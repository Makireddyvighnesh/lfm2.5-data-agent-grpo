"""Summarize recorded GRPO update and Harbor phase timings without touching services."""
import argparse
import gzip
import json
import statistics
from datetime import datetime
from pathlib import Path

def duration(phase):
    if not phase or not phase.get("finished_at"):
        return None
    return (datetime.fromisoformat(phase["finished_at"]) - datetime.fromisoformat(phase["started_at"])).total_seconds()

def summarize(values):
    if not values:
        return None
    v = sorted(values)
    return {"n": len(v), "mean": round(statistics.mean(v), 2), "median": round(statistics.median(v), 2),
            "p95": round(v[min(len(v)-1, int(.95*len(v)))], 2), "max": round(max(v), 2)}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--step", type=int)
    p.add_argument("--out")
    a = p.parse_args()
    trials = Path(__file__).resolve().parents[2] / "SmolDataEnvs/runs/trials_data_agent"
    phases = {k: [] for k in ["environment_setup", "agent_setup", "agent_execution", "verifier", "teardown_tail", "rollout"]}
    errors = {k: 0 for k in ["error", "capture_error", "cleanup_error"]}
    correct = timed_out = count = 0
    for f in Path(a.run, "rollouts").glob("*.json.gz"):
        m = json.load(gzip.open(f))["metrics"]
        if a.step and m.get("intended_step") != a.step:
            continue
        count += 1
        correct += m.get("correct") == 1
        for key in errors:
            errors[key] += bool(m.get(key))
        phases["rollout"].append(m["seconds"])
        path = trials / m.get("trial", "missing") / "result.json"
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        timed_out += bool(r.get("exception_info") and "timeout" in json.dumps(r["exception_info"]).lower())
        for key in ["environment_setup", "agent_setup", "agent_execution", "verifier"]:
            value = duration(r.get(key))
            if value is not None:
                phases[key].append(value)
        if (r.get("verifier") or {}).get("finished_at") and r.get("finished_at"):
            phases["teardown_tail"].append((datetime.fromisoformat(r["finished_at"]) -
                                            datetime.fromisoformat(r["verifier"]["finished_at"])).total_seconds())
    metrics = Path(a.run, "metrics.jsonl")
    updates = [json.loads(x) for x in metrics.read_text().splitlines()] if metrics.exists() else []
    if a.step:
        updates = [r for r in updates if r["step"] == a.step]
    result = {"run": a.run, "rollouts": count, "correct": correct, "timeouts": timed_out, "errors": errors,
              "phases_s": {k: summarize(v) for k, v in phases.items()},
              "updates": [{k:r.get(k) for k in ["step", "gen_s", "train_s", "sync_s", "step_s", "kl",
                          "ratio_mean", "tis_capped_frac", "forward_tokens", "trained_rows", "peak_mem_gib"]} for r in updates]}
    print(json.dumps(result, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=2) + "\n")

if __name__ == "__main__":
    main()
