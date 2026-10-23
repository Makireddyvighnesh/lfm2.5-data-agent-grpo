#!/usr/bin/env bash
# After the p8u200 val118 eval: test250 (pinned grader, 24k, r1) for p8u200 on pi / mini-swe-agent / codex, plus
# p7u350 on mini-swe-agent / codex for a same-harness baseline (p7u350 pi = 61.6 already), then epoch 2 (run_train_mh_e2.sh).
# usage: ./run_p8_pipeline.sh   (run in tmux; finished test250 results are skipped)
set -uo pipefail
DA="$(cd "$(dirname "$0")" && pwd)"; cd "$DA"
until grep -q "baseline done" runs/mh_eval_p8u200.out 2>/dev/null; do sleep 60; done
run() { local h=$1 label=$2 model=$3 adapter=$4
  [ -f "results/test250_${label}_r1.json" ] && { echo "skip $label"; return; }
  echo "$(date -u '+%F %T') test250 $label"; HARNESS=$h ./run_test250_p7.sh "$label" "$model" "$adapter"; }
run pi             mh24k_pi_p8u200    p8u200 "$DA/runs/p8_mh/checkpoint-200/adapter"
run mini-swe-agent mh24k_mini-swe-agent_p8u200 p8u200 "$DA/runs/p8_mh/checkpoint-200/adapter"
run codex          mh24k_codex_p8u200 p8u200 "$DA/runs/p8_mh/checkpoint-200/adapter"
run mini-swe-agent mh24k_mini-swe-agent_p7u350 p7u350 "$DA/runs/p7_think_bash/eval_adapters/u350"
run codex          mh24k_codex_p7u350 p7u350 "$DA/runs/p7_think_bash/eval_adapters/u350"
echo "$(date -u '+%F %T') test250 done; starting epoch 2"
./run_train_mh_e2.sh 120
echo "$(date -u '+%F %T') epoch 2 exited $?"
