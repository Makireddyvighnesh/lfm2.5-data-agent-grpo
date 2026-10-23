#!/usr/bin/env bash
# Fresh GRPO run (2026-10-13) on 500 train tasks with the fixes from the rollout review (FINDINGS.md):
#   - base LFM2.5-2.6B + new LoRA (no init adapter), Pi harness, FP8 sampler + vLLM-kernel FP8 trainer, TIS cap 2
#   - 8 tasks x 4 rollouts = 32 rollouts per update; one optimizer step per update over token-packed micro-batches
#     (gradient accumulation, 32k tokens per micro-batch); two passes over 500 tasks = 125 updates
#   - reward: blog tool reward, graded by grader_v3 (prepared_v3/datasets/train); std floor 0.1
#   - proxy: tool output capped at 4,000 chars; output reservation shrinks instead of a context-limit stop
#   - val118 (pinned grader, 4 rollouts/task) at updates 0, 25, 50, 75, 100, 125; v3 re-grade offline
# Run inside tmux. Stop: touch runs/p4_train500/STOP. Resume: rerun this script.
set -euo pipefail
DA="$(cd "$(dirname "$0")" && pwd)"
SMOL="$(cd "$DA/../SmolDataEnvs" && pwd)"
run=p4_train500
kv=${KV:-fp8}
budget=${BUDGET:-32768}
total=125
mkdir -p "$DA/runs/$run"
exec >> "$DA/runs/train500_pipeline.log" 2>&1
say() { echo "$(date -u '+%F %T UTC') $*"; }
harbor_ok() {  # healthy AND started with the v3 train split + tool-output cap
  local pid
  pid=$(ss -ltnp | grep -E ":8201 " | grep -oE "pid=[0-9]+" | head -1 | cut -d= -f2 || true)
  [ -n "$pid" ] && curl -sf --max-time 5 localhost:8201/health >/dev/null &&
    tr '\0' '\n' < "/proc/$pid/environ" | grep -q "prepared_v3/datasets/train" &&
    tr '\0' '\n' < "/proc/$pid/environ" | grep -q "^HARBOR_TOOL_OUTPUT_CAP=4000$"
}
ensure_services() {
  if ! harbor_ok; then
    pid=$(ss -ltnp | grep -E ":8201 " | grep -oE "pid=[0-9]+" | head -1 | cut -d= -f2 || true)
    if [ -n "$pid" ]; then kill "$pid"; for _ in $(seq 30); do ss -ltn | grep -qE ":8201 " || break; sleep 1; done; fi
    say "starting Harbor: v3-graded train500 split, tool-output cap 4000, output-reservation shrink"
    OPENENV_DATASETS="$DA/prepared/datasets/eval,$SMOL/prepared/datasets/test,$DA/prepared_v3/datasets/train" \
      HARBOR_TOOL_OUTPUT_CAP=4000 TRL_EXPERIMENTAL_SILENCE=1 setsid nohup "$SMOL/.venv/bin/python" -u \
      "$DA/serve_harbor.py" --port 8201 --capture-port 8101 --trials "$SMOL/runs/trials_data_agent" \
      >> "$DA/runs/harbor_p4.log" 2>&1 < /dev/null &
    for _ in $(seq 60); do harbor_ok && break; sleep 2; done
    harbor_ok
  fi
  if ! curl -sf --max-time 5 localhost:8000/health >/dev/null || \
     [ "$(cat "$DA/runs/vllm.mode" 2>/dev/null)" != "chain1_seqs64_kv$kv" ]; then
    CHAIN=1 SEQS=64 KV="$kv" "$DA/restart_vllm.sh"
  fi
  if curl -sf localhost:8000/is_sleeping | grep -q true; then
    if ! curl -sf -X POST localhost:8000/wake_up >/dev/null; then
      say "wake failed; restarting vLLM"
      CHAIN=1 SEQS=64 KV="$kv" "$DA/restart_vllm.sh"
    fi
  fi
}
stop_requested() { [ -f "$DA/runs/$run/STOP" ]; }
saved_step() {
  "$SMOL/.venv/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["step"])' \
    "$DA/runs/$run/state_latest/state.json" 2>/dev/null || echo 0
}
train_to() {
  local target=$1
  for attempt in 1 2 3 4; do
    stop_requested && { say "STOP requested"; return 2; }
    ensure_services
    say "training to update $target (attempt $attempt)"
    if TASKS=train500 CHAIN=1 CONC=32 TIMEOUT=300 "$DA/train_grpo.sh" "$run" "$target" \
      --sampler sequential --groups 8 --num-generations 4 --hard-generations 4 \
      --reward tool --std-floor 0.1 --lr 1e-5 \
      --fp8-base --fp8-mode vllm --is-correction tis --tis-cap 2 --token-budget "$budget" \
      --save-steps 25 --max-infra-fail 0.1 --data-root "$DA/prepared_v3" >> "$DA/runs/$run.log" 2>&1; then
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
  local step=$1 name="${run}-u$1" model adapter=""
  local result="$DA/results/val118_${name}_r4.json"
  if [ -f "$result" ] && "$SMOL/.venv/bin/python" -c \
    'import json,sys; r=json.load(open(sys.argv[1])); sys.exit(r.get("graded") != 472)' "$result"; then
    say "evaluation $name already complete"; return 0
  fi
  stop_requested && return 2
  ensure_services
  if [ "$step" = 0 ]; then
    model="LiquidAI/LFM2.5-2.6B"
  else
    model="$name"
    adapter="$DA/runs/$run/eval_adapters/u$step"
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
    "$SMOL/.venv/bin/python" -c 'import httpx,sys; httpx.post("http://127.0.0.1:8000/v1/load_lora_adapter",json={"lora_name":sys.argv[1],"lora_path":sys.argv[2]},timeout=120).raise_for_status()' "$name" "$adapter"
  fi
  say "val118 at update $step ($model), four rollouts/task (472 trials)"
  (cd "$SMOL"; PIRL_PREFIX_CHAIN=1 TRL_EXPERIMENTAL_SILENCE=1 .venv/bin/python -u pirl/evaluate.py \
    --label "$name" --model "$model" --split eval --data "$DA/prepared" --task-file "$DA/manifests/val118.json" \
    --rollouts-per-task 4 --concurrency 64 --agent-timeout 300 --server http://127.0.0.1:8201 \
    --vllm-url http://127.0.0.1:8000 --trials "$SMOL/runs/trials_data_agent" \
    --output "$DA/runs/val118_${name}_r4" --results-json "$result") \
    >> "$DA/runs/val118_${name}_r4.log" 2>&1
  "$SMOL/.venv/bin/python" - "$name" "$result" "$step" <<'PY'
import httpx,json,sys
r=json.load(open(sys.argv[2]));print('validation',sys.argv[1],'graded',r['graded'],'pass@1',r['pass_at_1'])
if sys.argv[3] != '0':
    httpx.post('http://127.0.0.1:8000/v1/unload_lora_adapter',json={'lora_name':sys.argv[1]},timeout=120).raise_for_status()
if r['graded'] != 472: raise SystemExit('incomplete validation; rerun supervisor to retry missing grades')
PY
}
say "starting $run: $total updates (2 passes x 500 tasks, 8 tasks x 4 rollouts), fresh LoRA on base"
evaluate_at 0
for target in 25 50 75 100 125; do
  train_to "$target"
  evaluate_at "$target"
done
say "completed $total updates and all scheduled validations"
touch "$DA/runs/$run/DONE"
