#!/usr/bin/env bash
# Claude Code test250 (r1, thinking OFF, turn cap 25, 600 s) of the p10 epoch-2 and epoch-1 adapters with a 32k vLLM
# window instead of 24k: at 24k ~19% of Claude Code rollouts run out of context (its prompt is ~10.8k tokens).
# Restarts vLLM at 32k (run_train_p10.sh restarts it at 24k on its next start). Finished results are skipped.
# usage: ./run_cc32k.sh   (run in tmux)
set -uo pipefail
DA="$(cd "$(dirname "$0")" && pwd)"; cd "$DA"
export PIRL_THINKING=0
mode=chain1_seqs96_kvfp8_NOTHINK_len32k_lr32
if [ "$(cat runs/vllm.mode 2>/dev/null)" != "$mode" ]; then
  echo "$(date -u '+%F %T UTC') restarting vLLM at 32k"
  MAX_LEN=32768 CHAIN=1 SEQS=96 KV=fp8 LORA_RANK=32 MAX_LORAS=6 ./restart_vllm.sh && echo "$mode" > runs/vllm.mode || exit 1
fi
for e in 2 1; do
  label=mh32k_claude-code_p10e$e
  [ -f "results/test250_${label}_r1.json" ] && { echo "skip $label"; continue; }
  echo "$(date -u '+%F %T UTC') test250 $label"
  HARNESS=claude-code ./run_test250_p7.sh "$label" "p10e$e" "$DA/runs/p10_mh_r32/epoch$e/adapter"
done
echo "$(date -u '+%F %T UTC') cc32k done"
