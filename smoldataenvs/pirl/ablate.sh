#!/usr/bin/env bash
# Reward ablation A/B/C on the L4 (LFM2.5-2.6B + LoRA, Pi harness, prefix-chain template).
#   train: same 100 medium+hard train prompts, 8 prompts x 4 rollouts per update, 25 updates (2 epochs)
#   eval:  base + A/B/C final adapters on the first 100 held-out medium+hard test tasks x 4 rollouts
# Resumable: rerunning skips finished trainings (state_latest at step 25) and finished eval rollouts.
# usage: pirl/ablate.sh            (run inside tmux; log: runs/ablate.log)
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
export TRL_EXPERIMENTAL_SILENCE=1
STEPS=${STEPS:-25}
log=runs/ablate.log
say() { echo "$(date -u +%FT%T) $*" | tee -a "$log"; }

for v in correctness calls efficiency; do
  out=runs/abl_$v
  say "train $v -> $out"
  pirl/run_resumable.sh "$out" --steps "$STEPS" --prompts-per-step 8 --num-generations 4 --concurrency 32 \
    --save-steps "$STEPS" --task-limit 100 --reward "$v"
  say "train $v done: $(cat $out/state_latest/state.json 2>/dev/null)"
done

say "eval: serving base + adapters (prefix-chain template, same as training)"
pirl/svc.sh stop vllm
args=()
for v in correctness calls efficiency; do args+=(--adapter "abl-$v=$PWD/runs/abl_$v/checkpoint-$STEPS/adapter"); done
pirl/svc.sh start vllm --output runs/vllm_ablate_eval --prefix-caching --gpu-mem 0.85 --max-num-seqs 64 --max-loras 4 "${args[@]}"
curl -sf localhost:8200/health >/dev/null || pirl/svc.sh start harbor
for label in base abl-correctness abl-calls abl-efficiency; do
  model=$([ "$label" = base ] && echo "LiquidAI/LFM2.5-2.6B" || echo "$label")
  say "eval $label"
  python pirl/evaluate.py --label "$label" --model "$model" --split test --difficulty medium,hard --tasks 100 \
    --rollouts-per-task 4 --concurrency 32 --output "runs/ablate_eval/$label" > "runs/ablate_eval_$label.log" 2>&1
done
python - <<'EOF' | tee -a "$log"
import json
from pathlib import Path
rows = {}
for d in sorted(Path("runs/ablate_eval").iterdir()):
    s = json.loads((d / "summary.json").read_text())
    rows[d.name] = {k: s[k] for k in ("graded", "pass_at_1", "submit_rate", "context_exhausted_rate", "mean_tool_calls",
                                      "mean_generated_tokens", "mean_seconds")}
    rows[d.name]["by_difficulty"] = {k: v["pass_at_1"] for k, v in s["difficulty"].items()}
Path("results/ablation_reward_100mh_x4.json").write_text(json.dumps(rows, indent=2))
print(f"{'model':18s} {'n':>4s} {'pass@1':>7s} {'submit':>7s} {'ctxEx':>6s} {'tools':>6s} {'genTok':>7s}")
for k, r in rows.items():
    print(f"{k:18s} {r['graded']:4d} {r['pass_at_1']:7.3f} {r['submit_rate']:7.3f} {r['context_exhausted_rate']:6.3f} "
          f"{r['mean_tool_calls']:6.2f} {r['mean_generated_tokens']:7.0f}")
EOF
say "ablation finished -> results/ablation_reward_100mh_x4.json"
