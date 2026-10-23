#!/usr/bin/env bash
# test250 (pinned grader), pi normal tools on Harbor :8201, 1 rollout/task, concurrency 16, agent timeout 600 s.
# usage: ./run_test250.sh <label> <model or lora name> [adapter dir]
set -euo pipefail
DA="$(cd "$(dirname "$0")" && pwd)"; SMOL="$(cd "$DA/../SmolDataEnvs" && pwd)"
label=$1; model=$2; adapter=${3:-}
curl -sf localhost:8000/is_sleeping | grep -q true && curl -sf -X POST localhost:8000/wake_up >/dev/null || true
[ -n "$adapter" ] && "$SMOL/.venv/bin/python" -c 'import httpx,sys; print(httpx.post("http://127.0.0.1:8000/v1/load_lora_adapter",json={"lora_name":sys.argv[1],"lora_path":sys.argv[2]},timeout=120).text[:120])' "$model" "$adapter"
(cd "$SMOL"; PIRL_PREFIX_CHAIN=1 TRL_EXPERIMENTAL_SILENCE=1 .venv/bin/python -u pirl/evaluate.py \
  --label "$label" --model "$model" --split test --tasks 250 --rollouts-per-task 1 --concurrency "${EVAL_CONC:-32}" --agent-timeout 600 \
  --server http://127.0.0.1:8201 --vllm-url http://127.0.0.1:8000 --trials "$SMOL/runs/trials_data_agent" \
  --output "$DA/runs/test250_${label}_r1" --results-json "$DA/results/test250_${label}_r1.json") >> "$DA/runs/test250_${label}_r1.log" 2>&1
"$SMOL/.venv/bin/python" -c 'import json,sys; r=json.load(open(sys.argv[1])); print("test250", sys.argv[2], "graded", r["graded"], "pass@1", r["pass_at_1"])' "$DA/results/test250_${label}_r1.json" "$label"
