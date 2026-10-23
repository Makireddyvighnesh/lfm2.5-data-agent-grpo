#!/usr/bin/env bash
# Two complete sequential passes; run this supervisor in dataagent-p2:train1000.
set -euo pipefail
DA="$(cd "$(dirname "$0")" && pwd)"
SMOL="$(cd "$DA/../SmolDataEnvs" && pwd)"
run=p3_train1000_tool
kv=${KV:-fp8}
budget=${BUDGET:-32768}
mkdir -p "$DA/runs/$run"
exec >> "$DA/runs/train1000_pipeline.log" 2>&1
say() { echo "$(date -u '+%F %T UTC') $*"; }
ensure_services() {
  if ! curl -sf --max-time 5 localhost:8201/health >/dev/null; then
    say "starting Harbor with cached inputs and prebuilt images"
    OPENENV_DATASETS="$DA/prepared/datasets/eval,$SMOL/prepared/datasets/test,$SMOL/prepared/datasets/train" \
      TRL_EXPERIMENTAL_SILENCE=1 setsid nohup "$SMOL/.venv/bin/python" -u "$DA/serve_harbor.py" \
      --port 8201 --capture-port 8101 --trials "$SMOL/runs/trials_data_agent" \
      >> "$DA/runs/harbor_codex.log" 2>&1 < /dev/null &
    for _ in $(seq 60); do curl -sf --max-time 5 localhost:8201/health >/dev/null && break; sleep 2; done
    curl -sf --max-time 5 localhost:8201/health >/dev/null
  fi
  if ! curl -sf --max-time 5 localhost:8000/health >/dev/null || \
     [ "$(cat "$DA/runs/vllm.mode" 2>/dev/null)" != "chain1_seqs64_kv$kv" ]; then
    CHAIN=1 SEQS=64 KV="$kv" "$DA/restart_vllm.sh"
  fi
  if curl -sf localhost:8000/is_sleeping | grep -q true; then
    if ! curl -sf -X POST localhost:8000/wake_up >/dev/null; then
      say "wake failed; restarting vLLM with the runtime worker fix"
      CHAIN=1 SEQS=64 KV="$kv" "$DA/restart_vllm.sh"
    fi
  fi
}
stop_requested() { [ -f "$DA/runs/$run/STOP" ]; }
train_to() {
  local target=$1
  for attempt in 1 2 3 4; do
    stop_requested && { say "STOP requested"; return 2; }
    ensure_services
    say "training to update $target (attempt $attempt, budget $budget, KV $kv)"
    if TASKS=train1000 CHAIN=1 CONC=64 TIMEOUT=240 "$DA/train_grpo.sh" "$run" "$target" \
      --sampler sequential --reward tool --std-floor 0.02 --lr 5e-6 --root-boost 1 \
      --fp8-base --fp8-mode vllm --is-correction tis --tis-cap 2 --token-budget "$budget" \
      --init-adapter "$DA/checkpoints/p3-init-A-tool" --save-steps 10 \
      --max-infra-fail 0.1 >> "$DA/runs/$run.log" 2>&1; then
      stop_requested && { say "STOP requested"; return 2; }
      return 0
    fi
    say "trainer exited; state_latest will resume the last saved update"
    sleep 10
  done
  say "four attempts failed; leaving checkpoint intact for inspection"
  return 1
}
evaluate_at() {
  local step=$1 name="${run}-u$1"
  local result="$DA/results/val118_${name}_r4.json"
  if [ -f "$result" ] && "$SMOL/.venv/bin/python" -c \
    'import json,sys; r=json.load(open(sys.argv[1])); sys.exit(r.get("graded") != 472)' "$result"; then
    say "evaluation $name already complete"; return 0
  fi
  stop_requested && return 2
  ensure_services
  local adapter="$DA/runs/$run/eval_adapters/u$step"
  if [ ! -d "$adapter" ]; then
    "$SMOL/.venv/bin/python" - "$DA/runs/$run" "$step" "$adapter" <<'PY'
import json,shutil,sys
from pathlib import Path
run=Path(sys.argv[1]);step=int(sys.argv[2]);dest=Path(sys.argv[3])
state=run/f'checkpoint-{step}'
if not state.exists(): state=run/'state_latest'
assert json.loads((state/'state.json').read_text())['step']==step, 'requested validation checkpoint is unavailable'
dest.parent.mkdir(parents=True,exist_ok=True)
shutil.copytree(state/'adapter',dest)
PY
  fi
  "$SMOL/.venv/bin/python" - "$name" "$adapter" <<'PY'
import httpx,sys
r=httpx.post('http://127.0.0.1:8000/v1/load_lora_adapter',json={'lora_name':sys.argv[1],'lora_path':sys.argv[2]},timeout=120)
r.raise_for_status()
PY
  say "val118 at update $step, four rollouts/task (472 trials)"
  (cd "$SMOL"; PIRL_PREFIX_CHAIN=1 TRL_EXPERIMENTAL_SILENCE=1 .venv/bin/python -u pirl/evaluate.py \
    --label "$name" --model "$name" --split eval --data "$DA/prepared" --task-file "$DA/manifests/val118.json" \
    --rollouts-per-task 4 --concurrency 64 --agent-timeout 300 --server http://127.0.0.1:8201 \
    --vllm-url http://127.0.0.1:8000 --trials "$SMOL/runs/trials_data_agent" \
    --output "$DA/runs/val118_${name}_r4" --results-json "$result") \
    >> "$DA/runs/val118_${name}_r4.log" 2>&1
  "$SMOL/.venv/bin/python" - "$name" "$result" <<'PY'
import httpx,json,sys
r=json.load(open(sys.argv[2]));print('validation',sys.argv[1],'graded',r['graded'],'pass@1',r['pass_at_1'])
httpx.post('http://127.0.0.1:8000/v1/unload_lora_adapter',json={'lora_name':sys.argv[1]},timeout=120).raise_for_status()
if r['graded'] != 472: raise SystemExit('incomplete validation; rerun supervisor to retry missing grades')
PY
}
say "starting $run: 250 updates, 1000 tasks, two passes, local GPU only"
for target in 50 100 125 175 225 250; do
  train_to "$target"
  evaluate_at "$target"
done
say "completed 250 updates and all scheduled validations"
touch "$DA/runs/$run/DONE"
