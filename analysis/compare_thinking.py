"""Compare two val118 eval runs on behaviour and accuracy (e.g. thinking off vs on).
usage: python3 analysis/compare_thinking.py <run dir A> <run dir B>
Per run: pass@1 (pinned), per-task paired difference, first-call tool, planning before the first action, read calls,
per-turn output tokens, turns cut at the 4,096 cap, peak context, context/length failures, seconds.
"""
import collections
import glob
import gzip
import json
import math
import statistics as st
import sys
from pathlib import Path


def text_of(c):
    return c if isinstance(c, str) else "".join(x.get("text", "") for x in (c or []) if isinstance(x, dict))


def load(run):
    run = Path(run)
    rows = [json.loads(l) for l in open(run / "rollouts.jsonl")]
    per_task = collections.defaultdict(list)
    m = collections.Counter()
    peaks, secs = [], []
    for r in rows:
        per_task[r["task"]].append(r.get("correctness") or 0)
        A = [x for x in r["trajectory"]["messages"] if x["role"] == "assistant"]
        calls = [tc["function"]["name"] for x in A for tc in (x.get("tool_calls") or [])]
        m["n"] += 1
        m["first_read"] += bool(calls) and calls[0] == "read"
        m["first_bash"] += bool(calls) and calls[0] == "bash"
        m["plan_first"] += bool(A) and len(text_of(A[0].get("content")).strip()) > 30
        m["reads"] += calls.count("read")
        m["ctx_ex"] += bool(r.get("context_exhausted"))
        m["submitted"] += bool(r.get("submitted"))
        peaks.append(r.get("peak_context") or 0)
        secs.append(r.get("seconds") or 0)
    lens = []
    for f in glob.glob(str(run / "captures/*.json.gz")):
        for t in json.load(gzip.open(f)).get("result", {}).get("turns") or []:
            if t.get("completion_token_ids"):
                lens.append(len(t["completion_token_ids"]))
    lens.sort()
    n = m["n"]
    s = {"attempts": n, "pass@1": sum(map(sum, per_task.values())) / n, "submitted": m["submitted"] / n,
         "first call = read": m["first_read"] / n, "first call = bash": m["first_bash"] / n,
         "NL plan before 1st action": m["plan_first"] / n, "read calls / attempt": m["reads"] / n,
         "output tokens / turn (median)": st.median(lens), "output tokens / turn (p90)": lens[int(0.9 * len(lens))],
         "turns cut at 4,096": sum(x >= 4096 for x in lens) / len(lens), "peak context (mean)": st.mean(peaks),
         "ran out of context": m["ctx_ex"] / n, "seconds / attempt": st.mean(secs)}
    return s, {t: sum(v) / len(v) for t, v in per_task.items()}


a, b = sys.argv[1], sys.argv[2]
sa, ta = load(a)
sb, tb = load(b)
print(f"{'':32} {Path(a).name[:26]:>26} {Path(b).name[:26]:>26}")
for k in sa:
    fa, fb = sa[k], sb[k]
    fmt = (lambda v: f"{v:.1%}") if isinstance(fa, float) and fa <= 1 and k not in ("read calls / attempt",) else (lambda v: f"{v:,.1f}" if isinstance(v, float) else f"{v:,}")
    print(f"{k:32} {fmt(fa):>26} {fmt(fb):>26}")
d = [tb[t] - ta[t] for t in ta if t in tb]
mu = st.mean(d)
z = mu / (st.stdev(d) / math.sqrt(len(d)))
print(f"\npaired over {len(d)} tasks: {mu * 100:+.1f} pts (z {z:.2f}); tasks better {sum(x > 0 for x in d)}, worse {sum(x < 0 for x in d)}")
