"""Audit every grader decision in the trial logs and flag likely grader failures.

Reads TRIALS/*/verifier/test-stdout.txt lines like
  [grader] gold='0.82' pred='78.43' reward=1.0 method=numeric_scaled
and re-checks each decision with a strict rule:
  numeric: |pred - gold| <= half a unit of the gold's last written decimal, or relative error <= 0.5%
           (percent/fraction bridge allowed only where the grader used it);
  labels:  case-, space-, underscore- and punctuation-insensitive equality;
  lists:   same multiset of elements under the two rules above.
False positive = grader accepted, strict rule rejects. False negative = grader rejected, strict rule accepts.
Containment (gold inside pred or pred inside gold) is reported separately as "review" (not counted as FN).

usage: python3 analysis/grader_audit.py            -> analysis/failure_review/grader_audit.{csv,md}
"""
import collections
import csv
import glob
import os
import re
from pathlib import Path

DA = Path(__file__).resolve().parents[1]
SMOL = DA.parent / "SmolDataEnvs"
TRIALS = SMOL / "runs" / "trials_data_agent"
SPLITS = {"train": SMOL / "prepared/datasets/train/tasks", "test": SMOL / "prepared/datasets/test/tasks",
          "eval": DA / "prepared/datasets/eval/tasks"}
OUT = DA / "analysis" / "failure_review"
LINE = re.compile(r"\[grader\] gold='(?P<gold>.*?)' pred='(?P<pred>.*)' (?:correctness|reward)=(?P<r>[01]\.0) method=(?P<m>\w+)")
NUM = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def split_of(task):
    for name, root in SPLITS.items():
        if (root / task).exists():
            return name
    return "?"


def question(task, split):
    p = SPLITS.get(split, Path("/nonexistent")) / task / "instruction.md"
    if not p.exists():
        return ""
    t = p.read_text()
    m = re.search(r"Question:\s*(.+?)\n\n", t, re.S)
    return (m.group(1) if m else "").strip().replace("\n", " ")[:200]


def num(s):
    s = (s or "").strip().strip("%$").replace(",", "")
    return float(s) if NUM.fullmatch(s) else None


def half_unit(x):
    t = x.strip().strip("%$").replace(",", "")
    if "e" in t.lower():
        mant = t.lower().split("e")[0]
        exp = int(t.lower().split("e")[1])
        d = len(mant.split(".")[1]) if "." in mant else 0
        return 0.5 * 10 ** (exp - d)
    return 0.5 * 10 ** -(len(t.split(".")[1]) if "." in t else 0)


def sig(x):
    t = x.strip().strip("%$").replace(",", "").lstrip("-").lower().split("e")[0].replace(".", "").lstrip("0")
    return len(t)


def same_number(gs, ps, scaled=False):
    """True if the two written numbers agree up to the precision of the less precise one (>= 3 sig digits)."""
    g, p = num(gs), num(ps)
    if g is None or p is None:
        return None
    for c in ([p, p * 100, p / 100] if scaled else [p]):
        if abs(c - g) <= half_unit(gs) * (1 + 1e-9):
            return True
        if sig(ps) >= 3 and abs(c - g) <= half_unit(ps) * (100 if c != p and False else 1) * (1 + 1e-9):
            return True
    return False


def norm(s):
    return re.sub(r"[^a-z0-9.\-]", "", (s or "").lower())


def numbers(s):
    return [float(x) for x in NUM.findall((s or "").replace(",", " "))]


def elem_ok(a, b):
    r = same_number(a, b)
    return r if r is not None else norm(a) == norm(b)


def list_ok(gold, pred):
    gl = [x.strip() for x in gold.split(",") if x.strip()]
    pl = [x.strip() for x in pred.split(",") if x.strip()]
    if len(gl) != len(pl):
        return False
    pool = list(pl)
    for a in gl:
        hit = next((b for b in pool if elem_ok(a, b)), None)
        if hit is None:
            return False
        pool.remove(hit)
    return True


def nums_match(gold, pred):
    """Same multiset of numbers (each within the gold's written precision or 0.1%)."""
    a, b = sorted(numbers(gold)), sorted(numbers(pred))
    if len(a) != len(b):
        return False
    return all(abs(x - y) <= max(1e-9, 0.001 * abs(x)) or abs(x - y) <= 0.5 for x, y in zip(a, b)) if a else False


def strict(gold, pred, method):
    if norm(gold) == norm(pred):
        return True
    if "," in gold or "," in pred:
        return list_ok(gold, pred)
    return bool(same_number(gold, pred, scaled=(method == "numeric_scaled")))


def rel_err(gold, pred, scaled=True):
    g, p = num(gold), num(pred)
    if g is None or p is None:
        return None
    best = min(abs(c - g) for c in ((p, p * 100, p / 100) if scaled else (p,)))
    return best / max(abs(g), 1e-300)


def classify_accept(gold, pred, meth):
    """Severity of an accepted answer that fails the strict check."""
    if meth in ("numeric", "numeric_scaled"):
        e = rel_err(gold, pred, scaled=(meth == "numeric_scaled"))
        if e is None:
            return "fp_unparsed"
        if e > 0.05 or (num(gold) or 0) * (num(pred) or 0) < 0:
            return "fp_wrong_value(>5%)"
        if e > 0.01:
            return "fp_loose_tolerance(1-5%)"
        return "tolerance_beyond_rounding(<=1%)"
    # list / math_verify: compare the numbers both strings contain
    if nums_match(gold, pred) and numbers(gold):
        return "lenient_same_numbers"
    if not numbers(gold) and not numbers(pred):
        return "fp_label_mismatch"
    return "fp_wrong_value"


def main():
    rows = collections.OrderedDict()
    n_lines = 0
    for f in glob.glob(str(TRIALS / "*" / "verifier" / "test-stdout.txt")):
        trial = Path(f).parents[1].name
        task = re.sub(r"^oe-", "", trial).rsplit("-", 1)[0]
        for line in open(f, errors="replace"):
            m = LINE.search(line)
            if not m:
                continue
            n_lines += 1
            gold, pred, ok, meth = m["gold"], m["pred"], m["r"] == "1.0", m["m"]
            s = strict(gold, pred, meth)
            if ok and not s:
                cat = classify_accept(gold, pred, meth)
                verdict = "false_positive" if cat.startswith("fp_") else "lenient_ok"
                cat = f"{meth}:{cat}"
            elif not ok and s:
                verdict, cat = "false_negative", ("format" if norm(gold) == norm(pred) else
                                                  "list_format" if ("," in gold or "," in pred) else "numeric_precision")
            elif not ok and len(norm(gold)) >= 3 and len(norm(pred)) >= 3 and (norm(gold) in norm(pred) or norm(pred) in norm(gold)):
                verdict, cat = "review", "containment"
            else:
                continue
            key = (task, gold, pred, meth, verdict)
            if key not in rows:
                sp = split_of(task)
                rows[key] = dict(task=task, split=sp, verdict=verdict, category=cat, method=meth, gold=gold, pred=pred,
                                 rel_err=("" if rel_err(gold, pred) is None else f"{rel_err(gold, pred):.3g}"), count=0, example_trial=trial, question=question(task, sp))
            rows[key]["count"] += 1
    data = sorted(rows.values(), key=lambda r: (r["verdict"], r["category"], -r["count"]))
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "grader_audit.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)
    summ = collections.Counter()
    for r in data:
        summ[(r["verdict"], r["category"], r["split"])] += r["count"]
    print("graded lines:", n_lines, "flagged unique cases:", len(data))
    for k, v in sorted(summ.items()):
        print(k, v)


if __name__ == "__main__":
    main()
