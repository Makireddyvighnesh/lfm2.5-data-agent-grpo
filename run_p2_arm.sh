#!/usr/bin/env bash
# Phase-2 reward ablation on 50 unseen hard tasks (manifests/hard50.json). Every arm starts from the same r2 adapter,
# with the same task order, template, sampling and limits. Only the reward differs.
#   A: --reward tool         blog recipe: correctness * (1 + 0.1*15/(15+calls))
#   B: --reward context      r2's: correctness * (1 + 0.1*clip((15k-peak)/5k,0,1))
#   C: --reward correctness  control
# Speed setup: prefix-chain template, 64 concurrent rollouts, 240 s agent timeout, fast teardown (serve_harbor.py).
# Needs: CHAIN=1 SEQS=64 ./restart_vllm.sh and serve_harbor.py on :8201. No root proxy (openings are about 99.5% efficient).
# usage: ./run_p2_arm.sh <A|B|C> [extra grpo args]
set -euo pipefail
DA="$(cd "$(dirname "$0")" && pwd)"
arm=$1; shift
case "$arm" in
  A) reward=tool ;;
  B) reward=context ;;
  C) reward=correctness ;;
  *) echo "arm must be A, B or C"; exit 1 ;;
esac
TASKS=hard50 CHAIN=1 CONC=64 TIMEOUT=240 exec "$DA/train_grpo.sh" "p2_${arm}_${reward}" 20 \
  --sampler sequential --reward "$reward" --std-floor 0.02 --root-boost 1 \
  --init-adapter "$DA/checkpoints/r2-final" --fp8-base --fp8-mode vllm --save-steps 10 "$@"
