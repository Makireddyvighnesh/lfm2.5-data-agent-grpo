#!/usr/bin/env bash
# GRPO in Pi on one L4 with the fixed Data Agent protocol (see PLAN.md).
# Needs: FP8 vLLM on :8000 (16k, --sleep-mode, LoRA r32) and serve_harbor.py on :8201 with the train split.
# usage: ./train_grpo.sh <run_name> <target_updates> [extra pirl/grpo.py args, e.g. --init-adapter DIR]
# Resumes automatically from runs/<run_name>/state_latest. Stop cleanly: touch runs/<run_name>/STOP
# Phase knobs (environment): TASKS (manifest, default train100), CHAIN (1 = prefix-chain template; vLLM must be
# serving the same template), CONC (concurrent rollouts, default 32), TIMEOUT (agent seconds, default 300).
set -euo pipefail
DA="$(cd "$(dirname "$0")" && pwd)"
SMOL="$(cd "$DA/../SmolDataEnvs" && pwd)"
run=$1; steps=$2; shift 2
cd "$SMOL"
source .venv/bin/activate
export TRL_EXPERIMENTAL_SILENCE=1 PIRL_PREFIX_CHAIN="${CHAIN:-0}"
exec python -u pirl/grpo.py --output "$DA/runs/$run" --steps "$steps" \
  --task-file "$DA/manifests/${TASKS:-train100}.json" \
  --groups 8 --num-generations 8 --hard-generations 8 --sampler regret --new-frac 0.5 \
  --lr 1e-5 --reward context --std-floor 0.1 --is-correction tis --tis-cap 2 \
  --token-budget 16384 --concurrency "${CONC:-32}" --agent-timeout "${TIMEOUT:-300}" --save-steps 25 \
  --server http://127.0.0.1:8201 --vllm-url http://127.0.0.1:8000 \
  --trials "$SMOL/runs/trials_data_agent" "$@"
