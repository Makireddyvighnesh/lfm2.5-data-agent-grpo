#!/usr/bin/env bash
# Short RL ablations from p7 u350 on abl40 (30 hard + 10 medium train tasks p7 solved <=1/8 on its last visit).
# Each arm: 10 updates x 4 tasks x 8 rollouts (one pass over the 40 tasks, same order), p7 setup
# (thinking ON, pi bash-only, turn cap 25, 600 s agent cap, FP8 + TIS, LoRA r16 init from p7 u350, fresh optimizer).
# Before/after: the same 40 tasks x 4 rollouts (grader v3.1), concurrency 16.
# usage: ./run_ablations.sh [arm ...]   (run in tmux; finished arms are skipped)
set -euo pipefail
DA="$(cd "$(dirname "$0")" && pwd)"; SMOL="$(cd "$DA/../SmolDataEnvs" && pwd)"
export PIRL_THINKING=1
INIT="$DA/runs/p7_think_bash/eval_adapters/u350"
exec >> "$DA/runs/ablations.log" 2>&1
say() { echo "$(date -u '+%F %T UTC') $*"; }
declare -A ARMS=(
  [base]=""                                   # p7 recipe unchanged
  [lr3e5]="--lr 3e-5"                         # 3x learning rate
  [binary]="--reward correctness"             # plain 0/1 reward, no speed/tool/token terms
  [cliphigh]="--epsilon-high 0.4"             # looser upper clip (DAPO clip-higher)
  [lr3e5_stream]="--lr 3e-5 --stream-groups 2"  # lr3e5 + 2 extra groups in flight (speed; run with CONC=48)
)
ORDER=(base lr3e5 binary cliphigh)
[ $# -gt 0 ] && ORDER=("$@")
evaluate() {  # <label> <lora name> <adapter dir>
  local label=$1 name=$2 adapter=$3 res="$DA/results/abl40_$1_r4.json"
  [ -f "$res" ] && { say "eval $label already done"; return 0; }
  curl -sf localhost:8000/is_sleeping | grep -q true && curl -sf -X POST localhost:8000/wake_up >/dev/null || true
  "$SMOL/.venv/bin/python" -c 'import httpx,sys; httpx.post("http://127.0.0.1:8000/v1/unload_lora_adapter",json={"lora_name":sys.argv[1]},timeout=60); print(httpx.post("http://127.0.0.1:8000/v1/load_lora_adapter",json={"lora_name":sys.argv[1],"lora_path":sys.argv[2]},timeout=120).text[:100])' "$name" "$adapter"
  say "eval $label: abl40 x4"
  (cd "$SMOL"; PIRL_PREFIX_CHAIN=1 TRL_EXPERIMENTAL_SILENCE=1 .venv/bin/python -u pirl/evaluate.py \
    --label "abl40_$label" --model "$name" --split train --data "$DA/prepared_v32" --task-file "$DA/manifests/abl40.json" --split-path "$DA/prepared_v32/datasets/train" \
    --tasks 40 --rollouts-per-task 4 --step-limit 25 --concurrency "${EVAL_CONC:-32}" --agent-timeout 600 \
    --server http://127.0.0.1:8201 --vllm-url http://127.0.0.1:8000 --trials "$SMOL/runs/trials_data_agent" \
    --output "$DA/runs/abl40_${label}_r4" --results-json "$res") >> "$DA/runs/abl40_${label}_r4.log" 2>&1
  "$SMOL/.venv/bin/python" -c 'import json,sys; r=json.load(open(sys.argv[1])); print("RESULT", sys.argv[2], "graded", r["graded"], "pass@1", r["pass_at_1"])' "$res" "$label"
  "$SMOL/.venv/bin/python" -c 'import httpx,sys; httpx.post("http://127.0.0.1:8000/v1/unload_lora_adapter",json={"lora_name":sys.argv[1]},timeout=60)' "$name"
}
evaluate p7u350 abl_p7u350 "$INIT"
for arm in "${ORDER[@]}"; do
  run="abl_$arm"
  if [ ! -f "$DA/runs/$run/state_latest/state.json" ] || ! grep -q '"step": 10' "$DA/runs/$run/state_latest/state.json"; then
    say "train $run (${ARMS[$arm]:-p7 recipe})"
    TASKS=abl40 CHAIN=1 CONC="${CONC:-32}" TIMEOUT=600 "$DA/train_grpo.sh" "$run" 10 \
      --sampler sequential --groups 4 --num-generations 8 --hard-generations 8 \
      --lora-r 16 --lora-alpha 32 --reward tool_fast --std-floor 0.1 --lr 1e-5 \
      --fp8-base --fp8-mode vllm --is-correction tis --tis-cap 2 --token-budget "${BUDGET:-32768}" \
      --save-steps 10 --max-infra-fail 0.1 --step-limit 25 --data-root "$DA/prepared_v32" \
      --init-adapter "$INIT" ${ARMS[$arm]} >> "$DA/runs/$run.log" 2>&1
  fi
  evaluate "$arm" "$run" "$DA/runs/$run/state_latest/adapter"
done
say "ablations done: ${ORDER[*]}"
