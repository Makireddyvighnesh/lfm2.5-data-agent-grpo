"""Read-only status for the p7_think_bash run: progress, GRPO metrics of recent updates, context-fix events, evals.
usage: python3 analysis/monitor_train1000f.py [--n 8]      (watch: watch -n 60 python3 analysis/monitor_train1000f.py)
"""
import argparse
import glob
import json
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

DA = Path(__file__).resolve().parents[1]
RUN = DA / "runs/p7_think_bash"
TOTAL = 400
p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=8)
a = p.parse_args()

print(datetime.now(timezone.utc).strftime("%F %T UTC"), "| p7_think_bash | 4 tasks x 8 rollouts, 500 updates, thinking ON, bash-only, turn cap 25, 798 tasks")
rows = [json.loads(x) for x in (RUN / "metrics.jsonl").read_text().splitlines()] if (RUN / "metrics.jsonl").exists() else []
if rows:
    last = rows[-1]
    print(f"update {last['step']}/{TOTAL}  epoch {min(2, 1 + (last['step'] - 1) // 200)}  task visits {last['cursor']}/798")
    hdr = ("upd  correct reward  r_std  adv_std |adv| zeroAdv  ent     kl     clip  tis   gnorm  ctxEx  peak  "
           "tools leak  callTxt  gen  train  total")
    print(hdr)
    for r in rows[-a.n:]:
        print(f"{r['step']:3} {r['correct_rate']:7.1%} {r['reward_mean']:6.3f} {r.get('reward_std', 0):6.3f} "
              f"{r.get('advantage_std', 0):7.3f} {r.get('advantage_abs_mean', 0):5.2f} {r['zero_adv_group_frac']:6.2f} "
              f"{r.get('entropy', 0):6.3f} {r.get('kl', 0):7.4f} {r.get('clip_frac', 0):5.1%} "
              f"{r.get('tis_capped_frac', 0):5.2%} {r.get('grad_norm', 0):6.3f} {r.get('context_exhausted_rate', 0):5.1%} "
              f"{r.get('peak_context_mean') or 0:6.0f} {r.get('tool_calls_mean') or 0:5.1f} "
              f"{r.get('think_leak_rate') or 0:5.0%} {r.get('ended_on_call_text_rate') or 0:6.1%} "
              f"{r['gen_s'] / 60:5.1f} {r.get('train_s', 0) / 60:5.1f} {r['step_s'] / 60:5.1f}")
    k = rows[-10:]
    mean = statistics.mean(r["step_s"] for r in k)
    print(f"group outcomes (last update, #correct of 8 -> groups): {last.get('group_correct_hist')}")
    print(f"mean {mean / 60:.1f} min/update; correct (last 10) {statistics.mean(r['correct_rate'] for r in k):.1%}; "
          f"training ETA {(datetime.now(timezone.utc) + timedelta(seconds=(TOTAL - last['step']) * mean)).strftime('%F %H:%M UTC')} (+ evals)")
else:
    print("no update finished yet")
# behaviour per update (from saved rollouts): first tool call `read`, and a bare number written for a label question
TRIALS = DA.parent / "SmolDataEnvs/runs/trials_data_agent"
NUM = __import__("re").compile(r"-?\d+(?:\.\d+)?")


def behaviour(step_rows):
    import gzip
    import re
    first_read = n = lab = numlab = 0
    for r in step_rows:
        try:
            d = json.load(gzip.open(RUN / r["capture_file"]))
            msgs = d["result"]["conversations"][0]["messages"]
        except Exception:
            continue
        calls = [tc["function"]["name"] for m in msgs if m.get("role") == "assistant" for tc in (m.get("tool_calls") or [])]
        if calls:
            n += 1
            first_read += calls[0] == "read"
        f = TRIALS / str(r.get("trial")) / "verifier" / "test-stdout.txt"
        m = re.search(r"gold='(.*?)' pred='(.*?)'", f.read_text(errors="replace")) if f.exists() else None
        if m and not NUM.search(m.group(1)):
            lab += 1
            numlab += bool(NUM.search(m.group(2))) and not re.search(r"[A-Za-z]{3,}", m.group(2))
    return (first_read / n if n else None), (numlab / lab if lab else None), lab


if rows and (RUN / "rollouts.jsonl").exists():
    by = {}
    for line in (RUN / "rollouts.jsonl").read_text().splitlines():
        x = json.loads(line)
        by.setdefault(x.get("step"), []).append(x)
    print("behaviour  upd  read-first  number-for-label (label questions)")
    for r in rows[-a.n:]:
        fr, nl, lab = behaviour(by.get(r["step"], []))
        print(f"           {r['step']:3}  {'-' if fr is None else f'{fr:6.0%}':>10}  "
              f"{'-' if nl is None else f'{nl:6.0%}':>8} ({lab})")
ev = DA / "runs/context_events.jsonl"
if ev.exists():
    since = rows[0]["completed_at"] - rows[0]["step_s"] if rows else 0
    c = {}
    for line in ev.read_text().splitlines():
        e = json.loads(line)
        if e["t"] >= since:
            c[e["event"]] = c.get(e["event"], 0) + 1
    print("proxy context events since run start:", c)
evals = sorted(glob.glob(str(DA / "results/val118_p7_think_bash-u*_r1.json")), key=lambda f: int(f.split("-u")[1].split("_")[0]))
if evals:
    print("val118 (pinned grader, 1/task):", ", ".join(
        f"u{f.split('-u')[1].split('_')[0]} {json.load(open(f))['pass_at_1']:.1%}" for f in evals))
log = DA / "runs/p7_pipeline.log"
if log.exists():
    print("supervisor:", log.read_text().splitlines()[-1])
for flag in ("STOP", "DONE"):
    if (RUN / flag).exists():
        print(flag)
