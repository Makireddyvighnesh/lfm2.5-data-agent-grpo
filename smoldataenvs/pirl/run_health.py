"""Check saved training captures and evaluations before extending a pilot run."""

import argparse
import gzip
import json
import math
from collections import Counter
from pathlib import Path

from pirl.artifacts import write_json


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def check_capture(run, row):
    errors = []
    filename = row.get("capture_file")
    if not filename:
        return ["missing capture_file"]
    path = (run / filename).resolve()
    if not path.is_relative_to(run.resolve()):
        return ["capture_file is outside the run directory"]
    try:
        with gzip.open(path, "rt") as stream:
            bundle = json.load(stream)
    except (OSError, ValueError) as exc:
        return [f"unreadable capture: {exc}"]
    if bundle["metrics"]["rollout_id"] != row["rollout_id"]:
        errors.append("capture belongs to another rollout")
    trace = bundle.get("training_trace")
    if row.get("reward") is not None and trace is None:
        errors.append("graded rollout has no archived training trace")
    for turn in (trace or {}).get("turns", []):
        prompt, completion = turn["prompt_token_ids"], turn["completion_token_ids"]
        logps, mask = turn["per_token_logps"], turn["loss_mask"]
        if len(logps) != len(completion) or len(mask) != len(prompt) + len(completion):
            errors.append("misaligned captured tokens/logprobs/mask")
        if any(not math.isfinite(v) or v > 1e-5 for v in logps) or any(v not in (0, 1) for v in mask):
            errors.append("invalid captured logprobs/mask")
    for seq in bundle.get("training_sequences", []):
        n = len(seq["input_ids"])
        if len(seq["old_log_probs"]) != n or len(seq["completion_mask"]) != n:
            errors.append("misaligned archived training sequence")
    return errors


def validate_run(run, expected_steps, baseline=None, candidate=None):
    run = Path(run)
    metrics = read_rows(run / "metrics.jsonl")
    rows = read_rows(run / "rollouts.jsonl")
    issues = []
    if [m["step"] for m in metrics] != list(range(1, expected_steps + 1)):
        issues.append(f"expected exactly {expected_steps} consecutive saved updates")
    state_path = run / "state_latest" / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if state.get("step") != expected_steps:
        issues.append("checkpoint does not match the pilot target")
    ids = [r.get("rollout_id") for r in rows]
    if None in ids or len(set(ids)) != len(ids):
        issues.append("missing or duplicate rollout IDs")
    counts = Counter(r["step"] for r in rows)
    for m in metrics:
        if counts[m["step"]] != m["rollouts"]:
            issues.append(f"step {m['step']}: rollout counts do not reconcile")
        for key in ("capture_errors", "infra_errors", "cleanup_errors", "openenv_loop_threads"):
            if key not in m or m[key] != 0:
                issues.append(f"step {m['step']}: {key} is missing or nonzero")
        for key in ("loss", "ratio_mean", "kl", "clip_frac", "grad_norm"):
            if not isinstance(m.get(key), (int, float)) or not math.isfinite(m[key]):
                issues.append(f"step {m['step']}: {key} is missing or non-finite")
        if m.get("ratio_mean") is not None and abs(m["ratio_mean"] - 1) > 0.02:
            issues.append(f"step {m['step']}: importance ratio differs from 1 by more than 0.02")
        if m.get("trained_completion_tokens", 0) <= 0 or m.get("grad_norm", 0) <= 0:
            issues.append(f"step {m['step']}: no effective gradient update")
    fds = [m["process_fds"] for m in metrics if "process_fds" in m]
    if len(fds) != len(metrics) or (fds and max(fds) - min(fds) > 64):
        issues.append("missing FD measurements or growth exceeding 64 descriptors")
    capture_errors = []
    for row in rows:
        errs = check_capture(run, row)
        if errs:
            capture_errors.append({"rollout_id": row.get("rollout_id"), "errors": errs})
    if capture_errors:
        issues.append(f"{len(capture_errors)} archived rollouts failed capture checks")
    log = run.parent / f"{run.name}.log"
    if log.exists() and "(restart #1)" in log.read_text():
        issues.append("pilot required an automatic trainer restart")
    evaluations = None
    if baseline is not None or candidate is not None:
        if baseline is None or candidate is None:
            issues.append("both baseline and candidate evaluations are required")
        else:
            base, trained = json.loads(Path(baseline).read_text()), json.loads(Path(candidate).read_text())
            keys = ("harness", "split", "instruction_hash", "difficulty_filter", "template",
                    "temperature", "top_p", "top_k", "max_tokens", "step_limit", "agent_timeout", "rollouts_per_task")
            if any(k not in base or k not in trained or base[k] != trained[k] for k in keys):
                issues.append("baseline and candidate evaluation settings/tasks differ")
            if not base.get("expected") or base.get("expected") != trained.get("expected"):
                issues.append("evaluation populations are empty or different")
            if base.get("coverage") != 1 or trained.get("coverage") != 1:
                issues.append("evaluation coverage is incomplete")
            for name, evaluation in (("baseline", base), ("candidate", trained)):
                if evaluation.get("trainable_rate") != 1 or evaluation.get("atif_match_rate") != 1:
                    issues.append(f"{name} evaluation has missing or failed token-capture validation")
            if base.get("pass_at_1") is None or trained.get("pass_at_1") is None:
                issues.append("evaluation accuracy is missing")
            elif trained["pass_at_1"] <= base["pass_at_1"]:
                issues.append("pilot has not improved observed held-out pass@1 over the matching baseline")
            evaluations = {"baseline": base.get("pass_at_1"), "candidate": trained.get("pass_at_1"),
                           "tasks": base.get("expected"), "template": base.get("template"),
                           "qualification": "An observed pilot improvement is not proof of statistical significance."}
    return {"passed": not issues, "expected_steps": expected_steps, "saved_updates": len(metrics),
            "saved_rollouts": len(rows), "issues": issues, "capture_errors": capture_errors,
            "evaluations": evaluations}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--candidate", type=Path)
    args = p.parse_args()
    report = validate_run(args.run, args.steps, args.baseline, args.candidate)
    write_json(args.run / "health.json", report)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
