"""val118 behaviour per eval: read-first, number-for-label, no-answer, context exhausted.
usage: python3 analysis/val_behaviour.py runs/val118_<label>_r4 [...]"""
import json, re, sys
from pathlib import Path
TRIALS = Path(__file__).resolve().parents[1].parent / "SmolDataEnvs/runs/trials_data_agent"
NUM = re.compile(r"-?\d+(?:\.\d+)?")
print(f"{'eval':38} {'n':>4} {'correct':>8} {'read1st':>8} {'num4lab':>12} {'noSubmit':>9} {'ctxEx':>6}")
for run in sys.argv[1:]:
    n = c = fr = calls_n = lab = nl = nosub = ctx = 0
    for line in open(Path(run) / "rollouts.jsonl"):
        x = json.loads(line); n += 1
        c += x.get("correctness") == 1.0
        nosub += not x.get("submitted"); ctx += bool(x.get("context_exhausted"))
        calls = [tc["function"]["name"] for m in x["trajectory"]["messages"] if m.get("role") == "assistant" for tc in (m.get("tool_calls") or [])]
        if calls: calls_n += 1; fr += calls[0] == "read"
        f = TRIALS / str(x.get("trial")) / "verifier" / "test-stdout.txt"
        m = re.search(r"gold='(.*?)' pred='(.*?)'", f.read_text(errors="replace")) if f.exists() else None
        if m and not NUM.search(m.group(1)):
            lab += 1; nl += bool(NUM.search(m.group(2))) and not re.search(r"[A-Za-z]{3,}", m.group(2))
    print(f"{Path(run).name:38} {n:4} {c/n:8.1%} {fr/max(calls_n,1):8.0%} {nl/max(lab,1):7.0%} ({lab:3}) {nosub/n:9.1%} {ctx/n:6.1%}")
