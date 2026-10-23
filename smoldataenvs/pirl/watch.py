"""Follow a GRPO run's metrics.jsonl as a table: python pirl/watch.py [runs/main200] [--once]"""

import json
import sys
import time
from pathlib import Path

COLS = [
    ("step", "step", "{:>4}"), ("prompt_passes", "visits", "{:>6.2f}"), ("reward_mean", "reward", "{:>6.3f}"),
    ("correct_rate", "correct", "{:>7.3f}"), ("zero_adv_group_frac", "zeroAdv", "{:>7.2f}"),
    ("submit_rate", "submit", "{:>6.2f}"), ("context_exhausted_rate", "ctxEx", "{:>5.2f}"),
    ("truncated_rate", "trunc", "{:>5.2f}"), ("tool_calls_mean", "tools", "{:>5.1f}"),
    ("generated_tokens_mean", "genTok", "{:>6.0f}"), ("ratio_mean", "ratio", "{:>6.4f}"),
    ("clip_frac", "clip", "{:>6.4f}"), ("grad_norm", "gnorm", "{:>5.2f}"),
    ("rollouts", "rolls", "{:>5}"), ("pool_frontier", "front", "{:>5}"), ("pool_dead", "dead", "{:>4}"),
    ("revived_prompts", "reviv", "{:>5}"),
    ("infra_errors", "err", "{:>3}"), ("gen_s", "gen_s", "{:>5.0f}"), ("train_s", "trn_s", "{:>5.0f}"),
    ("step_s", "step_s", "{:>6.0f}"), ("peak_mem_gib", "memGB", "{:>5.1f}"),
]


def fmt(rec):
    out = []
    for key, _, f in COLS:
        v = rec.get(key)
        out.append(f.format(v) if v is not None else f"{'-':>{len(f.format(0))}}")
    return " ".join(out)


def main():
    run = Path(next((a for a in sys.argv[1:] if not a.startswith("--")), "runs/main200"))
    path = run / "metrics.jsonl"
    header = " ".join(f"{name:>{len(f.format(0))}}" for _, name, f in COLS)
    print(header)
    seen, times = 0, []
    while True:
        lines = path.read_text().splitlines() if path.exists() else []
        for line in lines[seen:]:
            rec = json.loads(line)
            times.append(rec.get("step_s") or 0)
            print(fmt(rec), flush=True)
        seen = len(lines)
        if "--once" in sys.argv:
            break
        if times:
            cfg = json.loads((run / "run_config.json").read_text())
            left = cfg["steps"] - seen
            avg = sum(times[-10:]) / len(times[-10:])
            print(f"\r  {seen}/{cfg['steps']} updates, ~{left * avg / 3600:.1f} h left", end="", flush=True)
        time.sleep(20)


if __name__ == "__main__":
    main()
