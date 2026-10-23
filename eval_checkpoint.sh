#!/usr/bin/env bash
# Held-out check of a LoRA checkpoint under the fixed protocol (FP8 vLLM :8000, 16k, T=1, no root proxy).
# usage: [CHAIN=0|1] ./eval_checkpoint.sh <adapter_dir> <name> [split=val118|test250]   (CHAIN must match vLLM)
# Writes results/<split>_<name>.json and analysis/probe_<split>_<name>.json (natural first actions, 32/task).
# Run only while no trainer is using vLLM (it must be awake).
set -euo pipefail
DA="$(cd "$(dirname "$0")" && pwd)"
SMOL="$(cd "$DA/../SmolDataEnvs" && pwd)"
src=$1; name=$2; split=${3:-val118}
mkdir -p "$DA/checkpoints"
[ "$(realpath "$src")" = "$(realpath -m "$DA/checkpoints/$name")" ] || { rm -rf "$DA/checkpoints/$name"; cp -r "$src" "$DA/checkpoints/$name"; }
curl -sf localhost:8000/is_sleeping | grep -q false || { echo "vLLM is asleep or down"; exit 1; }
curl -s -X POST localhost:8000/v1/load_lora_adapter -H 'Content-Type: application/json' \
  -d "{\"lora_name\":\"$name\",\"lora_path\":\"$DA/checkpoints/$name\"}"; echo
cd "$SMOL"; source .venv/bin/activate; export TRL_EXPERIMENTAL_SILENCE=1 PIRL_PREFIX_CHAIN=${CHAIN:-0}
if [ "$split" = "val118" ]; then
  sel=(--split eval --data "$DA/prepared" --task-file "$DA/manifests/val118.json"); tdir="$DA/prepared/datasets/eval/tasks"; tf="$DA/manifests/val118.json"
else
  sel=(--split test --tasks 250); tdir="$SMOL/prepared/datasets/test/tasks"
  python -c "import json,os; json.dump(sorted(os.listdir('$tdir')), open('$DA/manifests/test250.json','w'))"; tf="$DA/manifests/test250.json"
fi
python pirl/evaluate.py --label "$name" --model "$name" "${sel[@]}" --concurrency 32 --server http://127.0.0.1:8201 \
  --trials "$SMOL/runs/trials_data_agent" --output "$DA/runs/${split}_$name" --results-json "$DA/results/${split}_$name.json" \
  > "$DA/runs/${split}_$name.log" 2>&1
cap=$(python3 -c "import glob,sys; print(sorted(glob.glob(sys.argv[1]))[0])" "$DA/runs/grpo_r2/rollouts/*.json.gz")
python -I "$DA/analysis/first_turn_probe.py" --capture "$cap" --tasks "$tf" --task-dir "$tdir" --n 32 --temperatures 1.0 \
  --model "$name" --out "$DA/analysis/probe_${split}_$name.json"
python -c "
import json; s=json.load(open('$DA/results/${split}_$name.json'))
print(f\"$name ${split}: pass@1 {s['pass_at_1']:.3f} ({round(s['pass_at_1']*s['graded'])}/{s['graded']}) submit {s['submit_rate']:.2f} ctx_out {s['context_exhausted_rate']:.2f} peak_ctx {s.get('mean_peak_context') or 0:.0f} tools {s['mean_tool_calls']:.1f}\")"
curl -s -X POST localhost:8000/v1/unload_lora_adapter -H 'Content-Type: application/json' -d "{\"lora_name\":\"$name\"}"; echo
