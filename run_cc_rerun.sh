#!/usr/bin/env bash
# Claude Code test250 r1 rerun (first try: 500/500 "Agent setup timed out" — Claude Code now prebaked in image pi-mh2).
# Waits for run_p9_confirm.sh to finish. usage: ./run_cc_rerun.sh
set -uo pipefail
DA="$(cd "$(dirname "$0")" && pwd)"; cd "$DA"
until grep -q "confirm done" runs/p9_confirm.out 2>/dev/null; do sleep 60; done
for m in "p9120 $DA/runs/p9_mh_e2/checkpoint-120/adapter" "p7u350 $DA/runs/p7_think_bash/eval_adapters/u350"; do
  set -- $m; [ -f "results/test250_mh24k_claude-code_$1_r1.json" ] && continue
  echo "$(date -u '+%F %T') test250 claude-code $1"; HARNESS=claude-code ./run_test250_p7.sh "mh24k_claude-code_$1" "$1" "$2"
done
echo "$(date -u '+%F %T') cc done"
