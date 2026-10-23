#!/usr/bin/env bash
# Multi-harness baseline: p7 u350 (trained on pi only) on val118 in each training harness, 24k window
# (vLLM MAX_LEN=24576, thinking ON, prefix chain), turn cap 25, 600 s, 1 rollout/task, concurrency 32.
# usage: ./run_mh_baseline.sh [harness ...]   (run in tmux; finished harnesses are skipped)
set -euo pipefail
DA="$(cd "$(dirname "$0")" && pwd)"; SMOL="$(cd "$DA/../SmolDataEnvs" && pwd)"
name=${NAME:-p7u350}; adapter="${ADAPTER:-$DA/runs/p7_think_bash/eval_adapters/u350}"
HS=("$@"); [ ${#HS[@]} -gt 0 ] || HS=(pi mini-swe-agent codex)
curl -sf localhost:8000/is_sleeping | grep -q true && curl -sf -X POST localhost:8000/wake_up >/dev/null || true
"$SMOL/.venv/bin/python" -c 'import httpx,sys; print(httpx.post("http://127.0.0.1:8000/v1/load_lora_adapter",json={"lora_name":sys.argv[1],"lora_path":sys.argv[2]},timeout=120).text[:120])' "$name" "$adapter"
for h in "${HS[@]}"; do
  res="$DA/results/val118_mh24k_${h}_${name}_r1.json"
  [ -f "$res" ] && { echo "skip $h"; continue; }
  echo "$(date -u '+%F %T') start $h"
  (cd "$SMOL"; PIRL_THINKING=1 PIRL_PREFIX_CHAIN=1 TRL_EXPERIMENTAL_SILENCE=1 .venv/bin/python -u pirl/evaluate.py \
    --label "mh24k_${h}_$name" --model "$name" --harness "$h" --split eval --data "$DA/prepared" --task-file "$DA/manifests/val118.json" \
    --rollouts-per-task 1 --step-limit 25 --concurrency "${EVAL_CONC:-32}" --agent-timeout 600 \
    --server http://127.0.0.1:8201 --vllm-url http://127.0.0.1:8000 --trials "$SMOL/runs/trials_data_agent" \
    --output "$DA/runs/val118_mh24k_${h}_${name}_r1" --results-json "$res") > "$DA/runs/val118_mh24k_${h}_${name}_r1.log" 2>&1
  "$SMOL/.venv/bin/python" -c 'import json,sys; s=json.load(open(sys.argv[1])); print(sys.argv[2], "graded", s["graded"], "pass@1", round(s["pass_at_1"],3), "ctx_out", round(s["context_exhausted_rate"],3), "peak", round(s.get("mean_peak_context") or 0), "tools", round(s["mean_tool_calls"],1))' "$res" "$h"
done
echo "$(date -u '+%F %T') baseline done"
