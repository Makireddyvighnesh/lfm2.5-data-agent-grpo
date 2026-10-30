#!/usr/bin/env bash
# Takes over the end of `EPOCH_STOP=3000 ./run_train_p10.sh` so the epoch-3 test250 runs in the user's order
# (held-out harnesses first). Polls every second for the trainer to exit, then at once kills the pipeline's bash
# (and any eval it already started, removing its partial output), checks epoch 3 finished, keeps the adapter in
# runs/p10_mh_r32/epoch3, then test250 r1 (24k, thinking OFF, turn cap 25, 600 s): opencode, claude-code, pi, mini, codex.
# usage: PIPE_PID=<pipeline bash> TRAIN_PID=<grpo.py> ./run_p10e3_eval.sh   (run in tmux)
set -uo pipefail
DA="$(cd "$(dirname "$0")" && pwd)"; cd "$DA"
export PIRL_THINKING=0
run=p10_mh_r32; ep="$DA/runs/$run/epoch3"
say() { echo "$(date -u '+%F %T UTC') $*"; }
while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 1; done
kill -9 "$PIPE_PID" 2>/dev/null
for pid in $(ps -eo pid,args | awk '/[r]un_test250_p7.sh .*p10e3|[p]irl\/evaluate.py .*p10e3/ {print $1}'); do kill -9 "$pid" 2>/dev/null; done
sleep 2
for d in runs/test250_mh24k_*_p10e3_r1; do
  [ -e "$d" ] && [ ! -f "results/$(basename "$d").json" ] && { say "removing partial $d"; rm -rf "$d" "$d.log"; }
done
grep -q "all task positions < 3000 are done" "runs/$run.log" || { say "trainer exited before the end of epoch 3"; exit 1; }
if [ ! -f "$ep/adapter/adapter_model.safetensors" ]; then
  mkdir -p "$ep" && cp -r "runs/$run/state_latest/adapter" "$ep/adapter" && cp "runs/$run/state_latest/state.json" "$ep/state.json"
fi
say "epoch 3 done at update $(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["step"])' "$ep/state.json"); adapter kept in $ep"
for h in opencode claude-code pi mini-swe-agent codex; do
  label=mh24k_${h}_p10e3
  [ -f "results/test250_${label}_r1.json" ] && { say "skip $label"; continue; }
  say "test250 $label"
  HARNESS=$h ./run_test250_p7.sh "$label" p10e3 "$ep/adapter"
done
say "epoch-3 eval done"
