#!/usr/bin/env bash
# Second independent OpenCode test250 run (r1, 24k, thinking OFF, turn cap 25, 600 s) of the p10 epoch-2 adapter,
# label mh24k_opencode_p10e2rep (the first run, test250_mh24k_opencode_p10e2_r1, is kept). Waits for run_cc32k.sh,
# then puts vLLM back at 24k.
# usage: ./run_oc24k_rep.sh   (run in tmux)
set -uo pipefail
DA="$(cd "$(dirname "$0")" && pwd)"; cd "$DA"
export PIRL_THINKING=0
until grep -q "cc32k done" runs/cc32k.log 2>/dev/null; do sleep 60; done
mode=chain1_seqs96_kvfp8_NOTHINK_len24k_lr32
if [ "$(cat runs/vllm.mode 2>/dev/null)" != "$mode" ]; then
  echo "$(date -u '+%F %T UTC') restarting vLLM at 24k"
  MAX_LEN=24576 CHAIN=1 SEQS=96 KV=fp8 LORA_RANK=32 MAX_LORAS=6 ./restart_vllm.sh && echo "$mode" > runs/vllm.mode || exit 1
fi
label=mh24k_opencode_p10e2rep
if [ ! -f "results/test250_${label}_r1.json" ]; then
  echo "$(date -u '+%F %T UTC') test250 $label"
  HARNESS=opencode ./run_test250_p7.sh "$label" p10e2 "$DA/runs/p10_mh_r32/epoch2/adapter"
fi
echo "$(date -u '+%F %T UTC') oc24k rep done"
