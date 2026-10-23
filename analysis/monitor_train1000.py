"""Readable, read-only status for the tmux-supervised 1000-task run."""
import json
import statistics
from datetime import datetime, timezone, timedelta
from pathlib import Path

DA = Path(__file__).resolve().parents[1]
run = DA / "runs/p3_train1000_tool"
print(datetime.now(timezone.utc).strftime('%F %T UTC'))
state = run / 'state_latest/state.json'
if state.exists():
    s = json.loads(state.read_text())
    print(f"Saved update {s['step']}/250; task visits {s['task_cursor']}/2000; unique tasks {min(s['task_cursor'],1000)}/1000")
else:
    print('Waiting for trainer initialization')
metrics = run / 'metrics.jsonl'
rows = [json.loads(x) for x in metrics.read_text().splitlines()] if metrics.exists() else []
if rows:
    print('\nupdate  gen(s) train(s) total(s) correct    KL      cap2  capture/infra')
    for r in rows[-5:]:
        print(f"{r['step']:6} {r['gen_s']:7.1f} {r.get('train_s',0):8.1f} {r['step_s']:8.1f} "
              f"{r['correct_rate']:7.1%} {r.get('kl',0):8.5f} {r.get('tis_capped_frac',0):7.2%} "
              f"{r['capture_errors']}/{r['infra_errors']}")
    mean = statistics.mean(r['step_s'] for r in rows[-10:])
    remaining = (250-rows[-1]['step'])*mean
    print(f"\nRecent mean {mean/60:.2f} min/update; remaining training {remaining/3600:.1f} h (validation adds time)")
    print('Training-only ETA', (datetime.now(timezone.utc)+timedelta(seconds=remaining)).strftime('%F %T UTC'))
log = DA / 'runs/train1000_pipeline.log'
if log.exists():
    print('\nSupervisor:')
    print('\n'.join(log.read_text().splitlines()[-3:]))
if (run/'STOP').exists(): print('\nSTOP marker is present')
if (run/'DONE').exists(): print('\nDONE')
