#!/usr/bin/env bash
# After epoch 2 (p9_mh_e2) exits: test250 r1, 24k, for the final checkpoint on pi / mini-swe-agent / codex (same setup as p8u200).
# usage: ./run_p9_eval.sh [checkpoint=checkpoint-120]   (run in tmux; finished results are skipped)
set -uo pipefail
DA="$(cd "$(dirname "$0")" && pwd)"; cd "$DA"
ck=${1:-checkpoint-120}; name=p9${ck#checkpoint-}
until grep -q "epoch 2 exited" runs/p8_pipeline.out 2>/dev/null; do sleep 60; done
[ -d "runs/p9_mh_e2/$ck/adapter" ] || { echo "missing runs/p9_mh_e2/$ck/adapter"; ls runs/p9_mh_e2; exit 1; }
for h in pi mini-swe-agent codex; do
  label=mh24k_${h}_$name
  [ -f "results/test250_${label}_r1.json" ] && { echo "skip $label"; continue; }
  echo "$(date -u '+%F %T') test250 $label"; HARNESS=$h ./run_test250_p7.sh "$label" "$name" "$DA/runs/p9_mh_e2/$ck/adapter"
done
echo "$(date -u '+%F %T') p9 eval done"
