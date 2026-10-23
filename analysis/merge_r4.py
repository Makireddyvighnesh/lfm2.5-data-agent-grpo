"""Merge test250 r1 + r3 runs into r4 pass@1 per harness/model, with a paired per-task comparison p9 vs p8."""
import json, math, sys
from pathlib import Path
RUNS = Path(__file__).resolve().parents[1] / "runs"

def per_task(h, name):
    d = {}
    for r in ("r1", "r3"):
        p = RUNS / f"test250_mh24k_{h}_{name}_{r}" / "rollouts.jsonl"
        if p.exists():
            for x in map(json.loads, open(p)): d.setdefault(x["task"], []).append(float(x.get("correctness") or 0))
    return d

for h in ("pi", "mini-swe-agent", "codex"):
    a, b = per_task(h, "p8u200"), per_task(h, "p9120")
    T = sorted(set(a) & set(b))
    pa = [sum(a[t]) / len(a[t]) for t in T]; pb = [sum(b[t]) / len(b[t]) for t in T]
    diff = [y - x for x, y in zip(pa, pb)]; m = sum(diff) / len(diff)
    se = math.sqrt(sum((v - m) ** 2 for v in diff) / (len(diff) - 1) / len(diff))
    n_a = sum(map(len, a.values())); n_b = sum(map(len, b.values()))
    print(f"{h:15} p8u200 {100*sum(pa)/len(pa):5.1f} ({n_a} att)  p9u120 {100*sum(pb)/len(pb):5.1f} ({n_b} att)  "
          f"diff {100*m:+5.1f} +- {100*1.96*se:4.1f} (95% CI, paired by task)")
