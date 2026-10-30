#!/usr/bin/env bash
# Restart the FP8 vLLM on :8000 with the Data Agent protocol. Never run while a trainer or evaluation is active.
# usage: CHAIN=0|1 SEQS=32|64|96 [KV=fp8] [MAX_LEN=24576] [LORA_RANK=64] [MAX_LORAS=4] ./restart_vllm.sh
#   CHAIN=1 serves the prefix-chain template (past assistant turns rendered with "<think></think>", so each
#   turn's prompt extends the previous prompt + completion and a rollout trains as one sequence).
set -euo pipefail
DA="$(cd "$(dirname "$0")" && pwd)"
SMOL="$(cd "$DA/../SmolDataEnvs" && pwd)"
chain=${CHAIN:-0}; seqs=${SEQS:-32}; kv=${KV:-auto}
pid=$(ss -ltnp | grep -E ":8000 " | grep -oE "pid=[0-9]+" | head -1 | cut -d= -f2 || true)
if [ -n "$pid" ]; then kill "$pid"; for _ in $(seq 120); do ss -ltn | grep -qE ":8000 " || break; sleep 1; done; sleep 5; fi
cd "$SMOL"; source .venv/bin/activate; export TRL_EXPERIMENTAL_SILENCE=1 PIRL_PREFIX_CHAIN=$chain
worker_args=()
if [ "$kv" = fp8 ]; then
  export PYTHONPATH="$DA${PYTHONPATH:+:$PYTHONPATH}"
  worker_args=(--worker-cls runtime_vllm_worker.PatchedWorker)
fi
out="$DA/runs/vllm_fp8_16k_chain${chain}_seqs${seqs}_kv${kv}"
setsid nohup python pirl/serve.py --output "$out" --gpu-mem 0.62 --max-num-seqs "$seqs" --max-model-len ${MAX_LEN:-16384} \
  --lora-rank ${LORA_RANK:-32} --max-loras ${MAX_LORAS:-4} --prefix-caching --sleep-mode --quantization fp8 ${KV:+--kv-cache-dtype $KV} "${worker_args[@]}" > "$out.log" 2>&1 < /dev/null &
for _ in $(seq 120); do curl -sf localhost:8000/health >/dev/null && break; sleep 5; done
curl -sf localhost:8000/health >/dev/null || { echo "vLLM failed to start; see $out.log"; exit 1; }
ss -ltnp | grep -E ":8000 " | grep -oE "pid=[0-9]+" | head -1 | cut -d= -f2 > "$DA/runs/vllm.pid"
echo "chain${chain}_seqs${seqs}_kv${kv}" > "$DA/runs/vllm.mode"
echo "vLLM up: chain=$chain max-num-seqs=$seqs pid=$(cat "$DA/runs/vllm.pid") template=$out/chat_template.jinja"
