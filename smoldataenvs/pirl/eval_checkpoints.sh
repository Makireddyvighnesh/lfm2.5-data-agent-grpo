#!/usr/bin/env bash
# Evaluate base + every checkpoint adapter of a run in one harness, under one chat template.
# usage: pirl/eval_checkpoints.sh <run_dir> <template: chain|native> <n_tasks> [harness=pi] [steps...]
# One vLLM serves the base model and all adapters (--lora-modules); results go to
# <run_dir>/eval_<template>_<harness>_<n>/<label>/summary.json and results/<run>_<template>_<harness>.json
set -euo pipefail
cd "$(dirname "$0")/.."
run=$1; template=$2; n=$3; harness=${4:-pi}; shift $(( $# >= 4 ? 4 : 3 ))
steps=("$@")
if [ ${#steps[@]} -eq 0 ]; then
  mapfile -t steps < <(ls -d "$run"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n)
fi
chain=$([ "$template" = native ] && echo 0 || echo 1)
adapters=()
for s in "${steps[@]}"; do adapters+=(--adapter "step-$s=$PWD/$run/checkpoint-$s/adapter"); done
pirl/svc.sh stop vllm
PIRL_PREFIX_CHAIN=$chain pirl/svc.sh start vllm --output "runs/vllm_eval_$template" --prefix-caching \
  --gpu-mem 0.85 --max-loras 4 "${adapters[@]}"
source .venv/bin/activate
export TRL_EXPERIMENTAL_SILENCE=1
out="$run/eval_${template}_${harness}_${n}"
name=$(basename "$run")
summary="{}"
for label in base "${steps[@]/#/step-}"; do
  model=$([ "$label" = base ] && echo "LiquidAI/LFM2.5-2.6B" || echo "$label")
  PIRL_PREFIX_CHAIN=$chain python pirl/evaluate.py --label "$label" --model "$model" --harness "$harness" \
    --split test --tasks "$n" --concurrency 16 --output "$out/$label" > "$out.$label.log" 2>&1 || true
done
python - "$out" "results/${name}_${template}_${harness}_${n}.json" <<'EOF'
import json, sys
from pathlib import Path
out, dest = Path(sys.argv[1]), Path(sys.argv[2])
rows = {}
for d in sorted(out.iterdir()):
    f = d / "summary.json"
    if f.exists():
        s = json.loads(f.read_text())
        rows[d.name] = {k: s[k] for k in ("graded", "pass_at_1", "submit_rate", "truncated_rate",
                        "context_exhausted_rate", "mean_tool_calls", "mean_generated_tokens", "mean_seconds")}
        rows[d.name]["by_difficulty"] = {k: v["pass_at_1"] for k, v in s["difficulty"].items()}
dest.parent.mkdir(exist_ok=True)
dest.write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=1))
EOF
