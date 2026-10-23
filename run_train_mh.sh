#!/usr/bin/env bash
# p8: multi-harness GRPO (pi, mini-swe-agent, codex; task index mod 3 picks the harness for all 8 rollouts),
# 24k window (vLLM MAX_LEN=24576), from p7 u350. Changes vs p7: harnesses, window, reward tool_ctx
# (correct: 1 + 0.1*15/(15+calls) + 0.1 if own context <= 15k; wrong: 0), lr 3e-5 cosine (warmup 10, min 3e-6),
# streaming rollouts (+2 groups in flight, concurrency 48). Everything else = p7 (thinking ON, prefix chain,
# FP8 + TIS cap 2, LoRA r16, groups 4 x 8, turn cap 25, 600 s, train798, sequential order).
# usage: ./run_train_mh.sh [updates=200]   (run in tmux; resumes from runs/p8_mh/state_latest; stop: touch runs/p8_mh/STOP)
set -euo pipefail
DA="$(cd "$(dirname "$0")" && pwd)"
run=p8_mh; steps=${1:-200}
grep -q len24k "$DA/runs/vllm.mode" || { echo "vLLM must run at 24k (MAX_LEN=24576 CHAIN=1 SEQS=64 KV=fp8 LORA_RANK=64 PIRL_THINKING=1 ./restart_vllm.sh)"; exit 1; }
init=(); [ -f "$DA/runs/$run/state_latest/state.json" ] || init=(--init-adapter "$DA/runs/p7_think_bash/eval_adapters/u350")
export PIRL_THINKING=1
TASKS=train798 CHAIN=1 CONC=48 TIMEOUT=600 "$DA/train_grpo.sh" "$run" "$steps" \
  --harnesses pi,mini-swe-agent,codex --sampler sequential --groups 4 --num-generations 8 --hard-generations 8 \
  --step-limit 25 --lora-r 16 --lora-alpha 32 --reward tool_ctx --std-floor 0.1 \
  --lr 3e-5 --lr-schedule cosine --warmup-steps 10 --min-lr 3e-6 --stream-groups 2 \
  --fp8-base --fp8-mode vllm --is-correction tis --tis-cap 2 --token-budget 32768 \
  --save-steps 25 --max-infra-fail 0.1 --data-root "$DA/prepared_v32" "${init[@]}" >> "$DA/runs/$run.log" 2>&1
