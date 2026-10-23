"""Read-only status for the p4_train500 run: progress, GRPO metrics of recent updates, context-fix events, evals.
usage: python3 analysis/monitor_train500.py [--n 8]      (watch: watch -n 60 python3 analysis/monitor_train500.py)
"""
import argparse
import glob
import json
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

DA = Path(__file__).resolve().parents[1]
RUN = DA / "runs/p4_train500"
TOTAL = 125
p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=8)
a = p.parse_args()

print(datetime.now(timezone.utc).strftime("%F %T UTC"), "| p4_train500 | 8 tasks x 4 rollouts, 125 updates")
rows = [json.loads(x) for x in (RUN / "metrics.jsonl").read_text().splitlines()] if (RUN / "metrics.jsonl").exists() else []
if rows:
    last = rows[-1]
    print(f"update {last['step']}/{TOTAL}  epoch {min(2, 1 + (last['step'] - 1) // 63)}  task visits {last['cursor']}/1000")
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
    print(f"group outcomes (last update, #correct of 4 -> groups): {last.get('group_correct_hist')}")
    print(f"mean {mean / 60:.1f} min/update; correct (last 10) {statistics.mean(r['correct_rate'] for r in k):.1%}; "
          f"training ETA {(datetime.now(timezone.utc) + timedelta(seconds=(TOTAL - last['step']) * mean)).strftime('%F %H:%M UTC')} (+ evals)")
else:
    print("no update finished yet")
ev = DA / "runs/context_events.jsonl"
if ev.exists():
    since = rows[0]["completed_at"] - rows[0]["step_s"] if rows else 0
    c = {}
    for line in ev.read_text().splitlines():
        e = json.loads(line)
        if e["t"] >= since:
            c[e["event"]] = c.get(e["event"], 0) + 1
    print("proxy context events since run start:", c)
evals = sorted(glob.glob(str(DA / "results/val118_p4_train500-u*_r4.json")), key=lambda f: int(f.split("-u")[1].split("_")[0]))
if evals:
    print("val118 (pinned grader, 4/task):", ", ".join(
        f"u{f.split('-u')[1].split('_')[0]} {json.load(open(f))['pass_at_1']:.1%}" for f in evals))
log = DA / "runs/train500_pipeline.log"
if log.exists():
    print("supervisor:", log.read_text().splitlines()[-1])
for flag in ("STOP", "DONE"):
    if (RUN / flag).exists():
        print(flag)
