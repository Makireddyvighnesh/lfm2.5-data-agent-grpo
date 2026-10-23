"""Stop one training supervisor and trainer, checking PID ownership before signalling."""

import argparse
import os
import signal
import time
from pathlib import Path


def matches(pid, script, run):
    try:
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
    except (FileNotFoundError, PermissionError):
        return False
    names = (script,) if isinstance(script, str) else script
    return any(Path(a).name in names for a in argv if a) and any(
        a and Path(a).resolve() == run for a in argv
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--force", action="store_true", help="Kill a matching process still alive after 15 seconds")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    run = args.run.resolve()
    if not run.is_relative_to(root / "runs") or not run.is_dir():
        p.error("run must be an existing directory under runs/")
    (run / "STOP").touch()
    pending = []
    invisible = []
    targets = (("pipeline", ("run_pilot.py", "pirl.run_pilot")),
               ("supervisor", "run_resumable.sh"), ("trainer", ("grpo.py", "pirl.grpo")))
    for name, script in targets:
        path = run / f"{name}.pid"
        if not path.exists():
            continue
        pid = int(path.read_text().strip())
        if not Path(f"/proc/{pid}").exists():
            invisible.append(pid)
            continue
        if not matches(pid, script, run):
            raise SystemExit(f"Refusing to signal PID {pid}: its command does not match {name} for {run}")
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        print(f"Sent SIGTERM to {name} PID {pid}", flush=True)
        pending.append((pid, script))
    deadline = time.monotonic() + 15
    while pending and time.monotonic() < deadline:
        pending = [(pid, script) for pid, script in pending if matches(pid, script, run)]
        time.sleep(0.2)
    if pending and args.force:
        for pid, script in pending:
            if matches(pid, script, run):
                os.kill(pid, signal.SIGKILL)
                print(f"Sent SIGKILL to {script} PID {pid}", flush=True)
    elif pending:
        raise SystemExit("Processes are still stopping; rerun with --force if needed.")
    if invisible and not pending:
        raise SystemExit(
            f"PID(s) {invisible} are not visible here. Run this command in the VM terminal; "
            "a PID namespace cannot confirm that host processes have stopped."
        )
    print(f"Stopped {run}; existing checkpoints and logs were preserved.", flush=True)


if __name__ == "__main__":
    main()
