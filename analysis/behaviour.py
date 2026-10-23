"""Per-update behaviour of a GRPO run: how the model starts each rollout and whether it fits the budget.

usage: python analysis/behaviour.py runs/<run>      (reads runs/<run>/rollouts/*.json.gz)
First call classes: full read (read without limit), read(limit), bash/python, other.
"""
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path


def first_call(turns):
    for t in turns:
        for c in t.get("tool_calls") or []:
            fn = c.get("function", c)
            name = fn.get("name")
            args = fn.get("arguments")
            args = args if isinstance(args, str) else json.dumps(args or {})
            if name == "read":
                return "read(limit)" if '"limit"' in args else "full read"
            if name == "bash":
                return "bash"
            return "other"
    return "no call"


def main(run):
    per = defaultdict(lambda: defaultdict(int))
    for f in Path(run, "rollouts").glob("*.json.gz"):
        d = json.load(gzip.open(f))
        m = d["metrics"]
        step = m.get("intended_step")
        p = per[step]
        p["n"] += 1
        p[first_call(d["result"]["turns"])] += 1
        p["correct"] += m.get("correct") == 1
        p["ctx"] += bool(m.get("context_exhausted"))
    cols = ["full read", "read(limit)", "bash", "other", "no call"]
    print(f"{'update':>6} {'n':>3} " + " ".join(f"{c:>11}" for c in cols) + f" {'correct':>8} {'out-of-ctx':>10}")
    for step in sorted(k for k in per if k is not None):
        p = per[step]
        print(f"{step:>6} {p['n']:>3} " + " ".join(f"{p[c] / p['n']:>11.0%}" for c in cols)
              + f" {p['correct'] / p['n']:>8.0%} {p['ctx'] / p['n']:>10.0%}")


if __name__ == "__main__":
    main(sys.argv[1])
