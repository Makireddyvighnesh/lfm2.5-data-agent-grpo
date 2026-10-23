"""Fixed short-answer grader (v3) for SmolDataEnvs tasks. Deterministic, no LLM judge, no network.

Drop-in alternative to the pinned tasks/*/tests/grader.py (sha256 16f906e9...), which stays untouched so
published scores remain comparable. Fixes the failures documented in analysis/failure_review/GRADER_FAILURES.md:

  pinned grader                                   v3
  ------------------------------------------------------------------------------------------------
  fixed ATOL (0.001 / 0.05) on any magnitude      no absolute tolerance; tolerance = the gold's written
                                                  precision (half a unit of its last digit) or the task RTOL
  zero tolerance on rounded integer golds         a pred that rounds to the gold at the gold's precision passes
  list tier compares the FIRST number of each     each element compares ALL its numbers plus its label text
  element ("Player1=50" is read as 1)
  math_verify fallback (partial / superset lists, removed; ratio golds "a:b" compared numerically
  "398, 9" == "392, 9"; pip-installed at grading)
  single-number tier reads the first number of a  a numeric gold needs exactly one number in the pred
  multi-value pred                                (or the same number repeated)
  labels compared with spaces / underscores       labels compared case-, space-, underscore- and
                                                  hyphen-insensitively
  pred more rounded than gold rejected            a pred with >= 3 significant digits may be rounded
  ("84.00" vs "83.9969")                          at its own precision

v3.1 (2026-10-13, from the blind judge audit of p4_train500 rollouts, analysis/verifier_p4/REPORT.md):
  - golds printed as "1.0" / "0.0" (float repr of a whole value) get +/-0.005, not +/-0.05 ("0.9667" != "1.0")
  - "True"/"False" answer "yes"/"no"; a trailing "regression" is a model-type word ("Ridge" == "Ridge regression")
  - "1.0" answers a percent gold "100"
The p4_train500 run's task copies (prepared_v3/datasets/train/tasks/*/tests/grader.py) remain v3.0.
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass

# Numbers only: digits glued to a name ("Player1", "PC2") are part of the label, not a value.
_NUM = re.compile(r"(?<![A-Za-z_\d.])-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![A-Za-z_])")


@dataclass
class GradeResult:
    reward: float
    method: str


_SUFFIX = re.compile(r"(classifier|regressor|regression model|regression|model)$")
_BOOL = {"true": "yes", "false": "no"}


def _key(s: str) -> str:
    """Label key: lower case, without spaces, underscores, quotes, trailing dots and a trailing model-type word
    ("RandomForestClassifier" == "Random Forest"). Hyphens are dropped only between letters, so a minus sign stays."""
    t = (s or "").strip().lower()
    t = re.sub(r"(?<=[a-z])-(?=[a-z])", "", t)
    t = re.sub(r"[\s_'\"`]+", "", t).rstrip(".")
    t = _BOOL.get(t, t)  # Python booleans answer yes/no questions
    stripped = _SUFFIX.sub("", t)
    return stripped or t


def _numbers(s: str) -> list[tuple[float, str]]:
    s = (s or "").replace("%", " ").replace("$", " ")
    return [(float(m.group(0)), m.group(0)) for m in _NUM.finditer(s)]


def _residue(s: str) -> str:
    """The text of an element with its numbers removed (its label part)."""
    return _key(_NUM.sub(" ", (s or "").replace("%", " ").replace("$", " ")).replace("=", " ").replace(":", " "))


def _half_unit(text: str) -> float:
    t = text.lstrip("-").lower()
    mant, _, exp = t.partition("e")
    decimals = len(mant.split(".")[1]) if "." in mant else 0
    return 0.5 * 10 ** ((int(exp) if exp else 0) - decimals)


def _sig_digits(text: str) -> int:
    mant = text.lstrip("-").lower().split("e")[0].replace(".", "").lstrip("0")
    return len(mant)


def _close(g: float, g_text: str, c: float, c_text: str, rel_tol: float) -> bool:
    diff = abs(c - g)
    rel_tol = max(rel_tol, 1e-6)  # floating-point noise (0.02879147907377122 vs ...123) always passes
    if g == 0 and "." not in g_text and "e" not in g_text.lower():
        tol = 1e-9  # exact zero ("0 missing values"); allow float noise only
    elif re.fullmatch(r"-?\d+\.0", g_text):
        tol = 0.005  # "1.0", "0.0", "25.0" are float reprs of a whole value, not a 1-decimal rounding
    else:
        tol = _half_unit(g_text)
    # strictly inside half a unit of the gold's last digit (0.95 does not round to 1.0 here), or within RTOL
    if diff < tol * (1 - 1e-9) or diff <= rel_tol * abs(g) * (1 + 1e-9):
        return True
    # the pred is more rounded than the gold but still precise (>= 3 significant digits)
    return "." in c_text and _sig_digits(c_text) >= 3 and diff < _half_unit(c_text) * (1 - 1e-9)


def _close_scaled(g, g_text, c, c_text, rel_tol) -> bool:
    if _close(g, g_text, c, c_text, rel_tol):
        return True
    # percent <-> fraction bridge, with the same strict tolerance
    if (0 < abs(c) < 1 <= abs(g)) or (0 < abs(g) < 1 <= abs(c)) or (abs(c) == 1 and "." in c_text and abs(g) == 100):
        for factor in (100.0, 0.01):
            scaled = c * factor
            if _close(g, g_text, scaled, repr(scaled), rel_tol):
                return True
    return False


def _ratio(s: str):
    m = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*:\s*(-?\d+(?:\.\d+)?)\s*", s or "")
    if m and float(m.group(2)) != 0:
        return float(m.group(1)) / float(m.group(2))
    return None


def _elem_match(g: str, c: str, rel_tol: float, in_list: bool) -> bool:
    if _key(g) == _key(c):
        return True
    gr, cr = _ratio(g), _ratio(c)
    if gr is not None:
        cn = _numbers(c)
        cv = cr if cr is not None else (cn[0][0] if len(cn) == 1 else None)
        return cv is not None and abs(cv - gr) <= max(0.005 * abs(gr), 1e-9)
    gn, cn = _numbers(g), _numbers(c)
    if not gn:
        return False  # label gold: only the label key comparison above counts
    if len(cn) != len(gn):
        # "50, 50, 50" for gold "50": accept a repeated identical number
        if len(gn) == 1 and cn and len({v for v, _ in cn}) == 1 and not in_list:
            cn = cn[:1]
        else:
            return False
    pairs = zip(sorted(gn), sorted(cn))
    check = _close_scaled if (len(gn) == 1 and not in_list) else _close
    if not all(check(gv, gt, cv, ct, rel_tol) for (gv, gt), (cv, ct) in pairs):
        return False
    rg, rc = _residue(g), _residue(c)
    # labels must agree; a bare number is fine; inside a list a shorter label contained in the longer is fine
    return rg == rc or not rc or not rg or (in_list and (rg in rc or rc in rg))


def _split_list(s: str, and_sep: bool = False) -> list[str]:
    parts = [p.strip() for p in re.split(r",|;" + (r"|\s+and\s+" if and_sep else ""), s or "", flags=re.I) if p.strip()]
    return [re.sub(r"^(and|&)\s+", "", p, flags=re.I) for p in parts]


def grade(gold: str, candidate: str, *, rel_tol: float = 0.0, **_ignored) -> GradeResult:
    gold, candidate = (gold or "").strip(), (candidate or "").strip()
    if not gold or not candidate:
        return GradeResult(0.0, "miss")
    if _key(gold) == _key(candidate):
        return GradeResult(1.0, "exact")
    if "," not in gold:
        # thousands separators in a single-value answer ("1,234")
        candidate = re.sub(r"(?<=\d),(?=\d{3}\b)", "", candidate)
    gl, cl = _split_list(gold), _split_list(candidate)
    if len(gl) == 1 and len(cl) >= 2:
        gl = _split_list(gold, and_sep=True)  # "carat and y" vs "carat, y"
    if len(gl) >= 2:
        if len(cl) != len(gl):
            return GradeResult(0.0, "miss")
        pool = list(cl)
        for g in gl:
            hit = next((c for c in pool if _elem_match(g, c, rel_tol, True)), None)
            if hit is None:
                return GradeResult(0.0, "miss")
            pool.remove(hit)
        return GradeResult(1.0, "list")
    if len(cl) >= 2 and len({_key(c) for c in cl}) > 1 and not _numbers(gold):
        return GradeResult(0.0, "miss")  # several different labels for one label gold
    if _elem_match(gold, candidate, rel_tol, False):
        return GradeResult(1.0, "numeric" if _numbers(gold) else "label")
    return GradeResult(0.0, "miss")


def main() -> int:
    """Same interface as the pinned grader: plain mode prints the reward; --json prints correctness/submission.
    The stderr line keeps the pinned "[grader] gold=... pred=..." shape (method prefixed "v3_") for log parsing."""
    import json
    gold = (os.environ.get("EXPECTED_ANSWER") or "").strip()
    candidate = sys.stdin.read().strip()
    try:
        rel = float(os.environ.get("RTOL", "") or 0.0)
    except ValueError:
        rel = 0.0
    if "--json" in sys.argv[1:] and not candidate:
        print(json.dumps({"correctness": 0.0, "submission": 0.0}))
        return 0
    r = grade(gold, candidate, rel_tol=rel)
    if "--json" in sys.argv[1:]:
        print(json.dumps({"correctness": float(r.reward), "submission": 1.0}))
    else:
        print(f"{r.reward:.1f}")
    print(f"[grader] gold={gold!r} pred={candidate[:80]!r} correctness={r.reward} method=v3_{r.method}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
