#!/usr/bin/env bash
# After run_train_p10.sh (EPOCH_STOP=1000) finishes its test250 on pi / mini-swe-agent / codex: test250 (r1, 24k,
# thinking OFF, turn cap 25, 600 s) of the p10 epoch-1 adapter on the two harnesses it never trained on,
# OpenCode and Claude Code (both prebaked in pirl/sandbox:pi-mh2).
# usage: ./run_p10e1_extra.sh   (run in tmux; finished harnesses are skipped)
set -uo pipefail
DA="$(cd "$(dirname "$0")" && pwd)"; cd "$DA"
export PIRL_THINKING=0
adapter="$DA/runs/p10_mh_r32/epoch1/adapter"
until grep -q "epoch-1 eval done" runs/p10_mh_r32_pipeline.log 2>/dev/null; do sleep 60; done
for h in opencode claude-code; do
  label=mh24k_${h}_p10e1
  [ -f "results/test250_${label}_r1.json" ] && { echo "skip $label"; continue; }
  echo "$(date -u '+%F %T UTC') test250 $label"
  HARNESS=$h ./run_test250_p7.sh "$label" p10e1 "$adapter"
done
echo "$(date -u '+%F %T UTC') extra harnesses done"
