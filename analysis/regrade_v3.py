"""Re-grade every logged answer with grader_v3 and compare with the pinned grader's verdict.

Inputs: SmolDataEnvs/runs/trials_data_agent/*/verifier/test-stdout.txt (gold, pred[:80], verdict, method) and each
task's RTOL from task.toml. Answers logged at the 80-char cut-off cannot be re-graded and keep the old verdict.
Outputs (analysis/failure_review/): regrade_v3.csv (one row per unique task/gold/pred with old and new verdicts) and
regrade_v3_trials.json (trial -> new correctness) for recomputing pass@1 per run.
usage: python3 analysis/regrade_v3.py [--strict]   (--strict ignores task RTOL: tolerance = gold rounding only)
"""
import ast
import collections
import csv
import glob
import json
import re
import sys
from pathlib import Path

DA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DA))
from grader_v3 import grade  # noqa: E402

SMOL = DA.parent / "SmolDataEnvs"
TRIALS = SMOL / "runs" / "trials_data_agent"
SPLITS = {"train": SMOL / "prepared/datasets/train/tasks", "test": SMOL / "prepared/datasets/test/tasks",
          "eval": DA / "prepared/datasets/eval/tasks"}
OUT = DA / "analysis" / "failure_review"
LINE = re.compile(r"\[grader\] gold=(?P<gold>'.*?'|\".*?\") pred=(?P<pred>'.*'|\".*\") "
                  r"(?:correctness|reward)=(?P<r>[01])\.0 method=(?P<m>\w+)")


def task_info(task):
    for split, root in SPLITS.items():
        toml = root / task / "task.toml"
        if toml.exists():
            s = toml.read_text()
            m = re.search(r'^RTOL = "([^"]*)"', s, re.M)
            q = (root / task / "instruction.md").read_text()
            qm = re.search(r"Question:\s*(.+?)\n\n", q, re.S)
            return split, float(m.group(1)) if m and m.group(1) else 1e-3, (qm.group(1).strip() if qm else "")
    return "?", 1e-3, ""


def main():
    strict = "--strict" in sys.argv[1:]
    suffix = "_strict" if strict else ""
    info, rows, trials = {}, collections.OrderedDict(), {}
    skipped = 0
    for f in glob.glob(str(TRIALS / "*" / "verifier" / "test-stdout.txt")):
        trial = Path(f).parents[1].name
        task = re.sub(r"^oe-", "", trial).rsplit("-", 1)[0]
        for line in open(f, errors="replace"):
            m = LINE.search(line)
            if not m:
                continue
            try:
                gold, pred = ast.literal_eval(m["gold"]), ast.literal_eval(m["pred"])
            except Exception:
                skipped += 1
                continue
            old = m["r"] == "1"
            if task not in info:
                info[task] = task_info(task)
            split, rtol, question = info[task]
            if len(pred) >= 80:  # logged prefix only
                new, meth = old, "kept_truncated"
            else:
                r = grade(gold, pred, rel_tol=0.0 if strict else rtol)
                new, meth = r.reward == 1.0, r.method
            trials[trial] = new
            key = (task, gold, pred)
            if key not in rows:
                rows[key] = dict(task=task, split=split, gold=gold, pred=pred, old=int(old), old_method=m["m"],
                                 new=int(new), new_method=meth, rtol=rtol, count=0, example_trial=trial, question=question[:200])
            rows[key]["count"] += 1
    data = list(rows.values())
    with open(OUT / f"regrade_v3{suffix}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(sorted(data, key=lambda r: (r["old"] == r["new"], -r["count"])))
    json.dump(trials, open(OUT / f"regrade_v3{suffix}_trials.json", "w"))
    c = collections.Counter()
    for r in data:
        c[(r["split"], r["old"], r["new"])] += r["count"]
    total = sum(r["count"] for r in data)
    print(f"graded answers {total}, unparsable log lines {skipped}, truncated kept "
          f"{sum(r['count'] for r in data if r['new_method'] == 'kept_truncated')}")
    for split in ("train", "eval", "test"):
        a = {k[1:]: v for k, v in c.items() if k[0] == split}
        n = sum(a.values())
        if n:
            print(f"{split:5} n={n:6d} old acc {sum(v for (o, _), v in a.items() if o)/n:.3f} "
                  f"new acc {sum(v for (_, nw), v in a.items() if nw)/n:.3f} | 1->0 {a.get((1, 0), 0)} 0->1 {a.get((0, 1), 0)}")


if __name__ == "__main__":
    main()
