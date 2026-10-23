#!/usr/bin/env bash
# Serve LiquidAI/LFM2.5-1.2B-Instruct on :8000 for evaluation: FP8, 16k, native chat template (instruct model, no <think>).
set -euo pipefail
DA="$(cd "$(dirname "$0")" && pwd)"; SMOL="$(cd "$DA/../SmolDataEnvs" && pwd)"
M=LiquidAI/LFM2.5-1.2B-Instruct; REV=0f604ada3f766f9f257460c4c9f0b5d6f69d431b
out="$DA/runs/vllm_lfm12b_fp8_16k"; mkdir -p "$out"
cd "$SMOL"; source .venv/bin/activate
setsid nohup vllm serve "$M" --revision "$REV" --host 0.0.0.0 --port 8000 --served-model-name "$M" \
  --dtype bfloat16 --max-model-len 16384 --max-num-seqs 64 --gpu-memory-utilization 0.62 \
  --enable-auto-tool-choice --tool-call-parser lfm2 --generation-config vllm \
  --logprobs-mode processed_logprobs --return-tokens-as-token-ids --enable-prefix-caching \
  --quantization fp8 > "$out.log" 2>&1 < /dev/null &
for _ in $(seq 120); do curl -sf localhost:8000/health >/dev/null && break; sleep 5; done
curl -sf localhost:8000/health >/dev/null || { echo "vLLM failed; see $out.log"; exit 1; }
echo "lfm12b_fp8" > "$DA/runs/vllm.mode"; echo "vLLM up: $M"
