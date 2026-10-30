#!/usr/bin/env bash
# Starts p11 epoch 2 (with the 2026-10-29 fixes in run_train_p11.sh) as soon as the epoch-1 test250 evals finish.
# usage: ./run_p11_chain.sh   (run in tmux; cancel: kill this tmux session before epoch 1's evals end)
set -uo pipefail
DA="$(cd "$(dirname "$0")" && pwd)"; cd "$DA"
until grep -qE "epoch-1 eval done|epoch-1 test250 failed|before the end of epoch 1" runs/p11_mh4_r32_pipeline.log 2>/dev/null; do sleep 60; done
grep -q "epoch-1 eval done" runs/p11_mh4_r32_pipeline.log || { echo "$(date -u '+%F %T UTC') epoch 1 did not finish cleanly; not starting epoch 2"; exit 1; }
while tmux has-session -t dataagent-p11 2>/dev/null; do sleep 10; done
echo "$(date -u '+%F %T UTC') starting epoch 2"
tmux new-session -d -s dataagent-p11 "EPOCH_STOP=2000 ./run_train_p11.sh"
