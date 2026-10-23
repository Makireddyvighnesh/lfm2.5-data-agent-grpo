#!/usr/bin/env bash
# p9: epoch 2 of p8_mh from runs/p8_mh/checkpoint-200. Changes vs p8: tasks = train_e2 (477 of train798 that were not
# solved 8/8 in epoch 1; never-solved kept since the harness rotation changes), lr 2e-5 cosine (warmup 5, min 2e-6).
# Everything else = p8 (pi, mini-swe-agent, codex; reward tool_ctx; 24k; 4 x 8; FP8 + TIS cap 2; LoRA r16).
# usage: ./run_train_mh_e2.sh [updates=120]   (run in tmux; resumes from runs/p9_mh_e2/state_latest; stop: touch runs/p9_mh_e2/STOP)
set -euo pipefail
DA="$(cd "$(dirname "$0")" && pwd)"
run=p9_mh_e2; steps=${1:-120}
grep -q len24k "$DA/runs/vllm.mode" || { echo "vLLM must run at 24k (MAX_LEN=24576 CHAIN=1 SEQS=64 KV=fp8 LORA_RANK=64 PIRL_THINKING=1 ./restart_vllm.sh)"; exit 1; }
init=(); [ -f "$DA/runs/$run/state_latest/state.json" ] || init=(--init-adapter "$DA/runs/p8_mh/checkpoint-200/adapter")
export PIRL_THINKING=1
TASKS=train_e2 CHAIN=1 CONC=48 TIMEOUT=600 "$DA/train_grpo.sh" "$run" "$steps" \
  --harnesses pi,mini-swe-agent,codex --sampler sequential --groups 4 --num-generations 8 --hard-generations 8 \
  --step-limit 25 --lora-r 16 --lora-alpha 32 --reward tool_ctx --std-floor 0.1 \
  --lr 2e-5 --lr-schedule cosine --warmup-steps 5 --min-lr 2e-6 --stream-groups 2 \
  --fp8-base --fp8-mode vllm --is-correction tis --tis-cap 2 --token-budget 32768 \
  --save-steps 20 --max-infra-fail 0.1 --data-root "$DA/prepared_v32" "${init[@]}" >> "$DA/runs/$run.log" 2>&1
