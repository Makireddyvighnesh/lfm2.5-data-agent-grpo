"""Start fresh for 50 updates; evaluate and extend to 200 only after pilot checks pass."""

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path

from pirl.artifacts import write_json
from pirl.common import ROOT
from pirl.run_health import validate_run
from pirl.stop_run import matches


def execute(argv, env):
    print("RUN " + shlex.join(map(str, argv)), flush=True)
    command = list(map(str, argv))
    child = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True)
    try:
        code = child.wait()
    except BaseException:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        raise
    if code:
        raise subprocess.CalledProcessError(code, command)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("runs/l4_fixed50"))
    p.add_argument("--extend-to", type=int, choices=(50, 200), default=200)
    p.add_argument("--eval-tasks", type=int, default=50)
    args = p.parse_args()
    os.chdir(ROOT)
    out = args.output.resolve()
    if not out.is_relative_to(ROOT / "runs"):
        p.error("output must be under runs/")
    if args.eval_tasks < 1 or args.eval_tasks > 217:
        p.error("eval-tasks must be between 1 and 217")
    if out.exists():
        raise SystemExit("Fresh pilot output already exists; choose a new --output directory")
    old = ROOT / "runs/l4_regret200"
    for name, script in (("supervisor", "run_resumable.sh"), ("trainer", "grpo.py")):
        pidfile = old / f"{name}.pid"
        if pidfile.exists() and matches(int(pidfile.read_text()), script, old):
            raise SystemExit("Old run is still active. First run: python3 pirl/stop_run.py runs/l4_regret200")
    env = {**os.environ, "PIRL_PREFIX_CHAIN": "1", "PIRL_MAX_RESTARTS": "0", "PYTHONUNBUFFERED": "1"}
    env["PYTHONPATH"] = str(ROOT / "OpenEnv/envs") + os.pathsep + env.get("PYTHONPATH", "")
    # This also fails clearly in a sandbox without the VM's GPU driver.
    try:
        execute(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"], env)
    except subprocess.CalledProcessError as exc:
        raise SystemExit("GPU preflight failed. Run this launcher in the VM terminal with GPU access; no services were changed.") from exc
    out.mkdir(parents=True)
    (out / "pipeline.pid").write_text(str(os.getpid()))
    write_json(out / "plan.json", {"pilot_steps": 50, "extend_to": args.extend_to,
               "eval_tasks": args.eval_tasks, "template": "prefix-chain", "reward": "efficiency",
               "extension_rule": "All capture, optimizer and resource checks pass; matching held-out pass@1 improves."})
    def interrupted(signum, frame):
        raise KeyboardInterrupt("pilot stop requested")

    previous_handler = signal.signal(signal.SIGTERM, interrupted)
    try:
        execute(["bash", "pirl/svc.sh", "stop", "harbor"], env)
        execute(["bash", "pirl/svc.sh", "start", "harbor"], env)
        execute(["bash", "pirl/svc.sh", "stop", "vllm"], env)
        execute(["bash", "pirl/svc.sh", "start", "vllm", "--output", out / "vllm",
                 "--prefix-caching", "--sleep-mode", "--gpu-mem", "0.62"], env)
        evaluation = [sys.executable, "-m", "pirl.evaluate", "--difficulty", "medium,hard",
                      "--tasks", str(args.eval_tasks), "--concurrency", "32", "--agent-timeout", "300"]
        execute(evaluation + ["--label", "base", "--output", out / "eval_base"], env)
        training = ["bash", "pirl/run_resumable.sh", out, "--groups", "8", "--num-generations", "4",
                    "--concurrency", "32", "--sampler", "regret", "--hard-generations", "8",
                    "--new-frac", "0.5", "--reward", "efficiency", "--save-steps", "10"]
        execute(training + ["--steps", "50"], env)
        health = validate_run(out, 50)
        write_json(out / "health.json", health)
        if not health["passed"]:
            raise RuntimeError("Pilot health checks failed: " + "; ".join(health["issues"]))
        execute(evaluation + ["--model", "trl-policy-v50", "--label", "step-50",
                              "--output", out / "eval_step50"], env)
        health = validate_run(out, 50, out / "eval_base/summary.json", out / "eval_step50/summary.json")
        write_json(out / "health.json", health)
        print(json.dumps(health, indent=2), flush=True)
        if not health["passed"]:
            raise RuntimeError("Pilot evaluation gate failed; keeping the run at 50 updates")
        if args.extend_to == 200:
            print("Pilot checks passed; resuming the same policy and optimizer for 200 total updates.", flush=True)
            execute(training + ["--steps", "200"], env)
            execute(evaluation + ["--model", "trl-policy-v200", "--label", "step-200",
                                  "--output", out / "eval_step200"], env)
        write_json(out / "pipeline.json", {"status": "complete", "completed_steps": args.extend_to})
    except (subprocess.CalledProcessError, RuntimeError, KeyboardInterrupt) as exc:
        write_json(out / "pipeline.json", {"status": "stopped", "reason": str(exc)})
        raise SystemExit(str(exc)) from exc
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


if __name__ == "__main__":
    main()
