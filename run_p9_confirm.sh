#!/usr/bin/env bash
# (1) Blog harnesses not trained on: test250 r1 on opencode / claude-code for p9 u120 and p7 u350 (transfer check).
# (2) r4 confirmation: 3 more attempts/task (RPT=3) on pi / mini-swe-agent / codex for p9 u120 and p8 u200;
#     merged with the existing r1 runs by analysis/merge_r4.py -> 4 attempts/task.
# Same setup as all p8/p9 test250 runs: 24k, thinking ON, turn cap 25, 600 s, concurrency 32.
# usage: ./run_p9_confirm.sh   (run in tmux; finished results are skipped)
set -uo pipefail
DA="$(cd "$(dirname "$0")" && pwd)"; cd "$DA"
P9="$DA/runs/p9_mh_e2/checkpoint-120/adapter"; P8="$DA/runs/p8_mh/checkpoint-200/adapter"; P7="$DA/runs/p7_think_bash/eval_adapters/u350"
run() { local rpt=$1 h=$2 name=$3 adapter=$4 label=mh24k_$2_$3
  [ -f "results/test250_${label}_r${rpt}.json" ] && { echo "skip $label r$rpt"; return; }
  echo "$(date -u '+%F %T') test250 $label r$rpt"; RPT=$rpt HARNESS=$h ./run_test250_p7.sh "$label" "$name" "$adapter"; }
for h in opencode claude-code; do run 1 $h p9120 "$P9"; run 1 $h p7u350 "$P7"; done
for h in pi mini-swe-agent codex; do run 3 $h p9120 "$P9"; run 3 $h p8u200 "$P8"; done
echo "$(date -u '+%F %T') confirm done"
