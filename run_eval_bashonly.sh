#!/usr/bin/env bash
# val118 with pi restricted to the bash tool only (no read/edit/write): python3 runs through bash.
# Same protocol as the p6 evals otherwise: pinned grader, 1 rollout/task, concurrency 16, agent timeout 600 s,
# 4,000-char tool-output cap, thinking off. Uses its own Harbor on :8202 (capture :8102); :8201 is untouched.
# usage: ./run_eval_bashonly.sh <label> <model or lora name> [adapter dir]   (run in tmux)
set -euo pipefail
DA="$(cd "$(dirname "$0")" && pwd)"
SMOL="$(cd "$DA/../SmolDataEnvs" && pwd)"
label=$1; model=$2; adapter=${3:-}
exec >> "$DA/runs/eval_bashonly.log" 2>&1
say() { echo "$(date -u '+%F %T UTC') $*"; }
harbor_ok() {
  local pid
  pid=$(ss -ltnp | grep -E ":8202 " | grep -oE "pid=[0-9]+" | head -1 | cut -d= -f2 || true)
  [ -n "$pid" ] && curl -sf --max-time 5 localhost:8202/health >/dev/null &&
    tr '\0' '\n' < "/proc/$pid/environ" | grep -q "^HARBOR_PI_TOOLS=bash$"
}
if ! harbor_ok; then
  say "starting bash-only Harbor on :8202"
  OPENENV_DATASETS="$DA/prepared/datasets/eval,$SMOL/prepared/datasets/test,$DA/prepared_v32/datasets/train" \
    HARBOR_TOOL_OUTPUT_CAP=4000 HARBOR_PI_TOOLS=bash TRL_EXPERIMENTAL_SILENCE=1 setsid nohup "$SMOL/.venv/bin/python" -u \
    "$DA/serve_harbor.py" --port 8202 --capture-port 8102 --trials "$SMOL/runs/trials_data_agent" \
    >> "$DA/runs/harbor_bashonly.log" 2>&1 < /dev/null &
  for _ in $(seq 60); do harbor_ok && break; sleep 2; done
  harbor_ok
fi
curl -sf localhost:8000/is_sleeping | grep -q true && curl -sf -X POST localhost:8000/wake_up >/dev/null
if [ -n "$adapter" ]; then
  "$SMOL/.venv/bin/python" -c 'import httpx,sys; r=httpx.post("http://127.0.0.1:8000/v1/load_lora_adapter",json={"lora_name":sys.argv[1],"lora_path":sys.argv[2]},timeout=120); print(r.status_code, r.text[:200])' "$model" "$adapter"
fi
result="$DA/results/val118_${label}_r1.json"
say "val118 bash-only: $label ($model), one rollout/task, concurrency 16"
(cd "$SMOL"; PIRL_PREFIX_CHAIN=1 TRL_EXPERIMENTAL_SILENCE=1 .venv/bin/python -u pirl/evaluate.py \
  --label "$label" --model "$model" --split eval --data "$DA/prepared" --task-file "$DA/manifests/val118.json" \
  --rollouts-per-task 1 --concurrency 16 --agent-timeout 600 --server http://127.0.0.1:8202 \
  --vllm-url http://127.0.0.1:8000 --trials "$SMOL/runs/trials_data_agent" \
  --output "$DA/runs/val118_${label}_r1" --results-json "$result") >> "$DA/runs/val118_${label}_r1.log" 2>&1
"$SMOL/.venv/bin/python" -c 'import json,sys; r=json.load(open(sys.argv[1])); print("bash-only", sys.argv[2], "graded", r["graded"], "pass@1", r["pass_at_1"])' "$result" "$label"
if [ -n "$adapter" ]; then
  "$SMOL/.venv/bin/python" -c 'import httpx,sys; httpx.post("http://127.0.0.1:8000/v1/unload_lora_adapter",json={"lora_name":sys.argv[1]},timeout=120)' "$model"
fi
