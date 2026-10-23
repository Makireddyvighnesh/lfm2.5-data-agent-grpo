#!/usr/bin/env bash
# p7_think_bash: p6 recipe + thinking ON + pi bash-only (--tools bash) + turn cap 25 + pool minus 202 tasks never solved in p6 (manifests/train798.json).
#   - 4 tasks x 8 rollouts = 32 rollouts per update; 500 updates (2 x 1000 / 4); micro-batches of --token-budget
#     tokens accumulate into one optimizer step (gradient accumulation), so OOM -> lower BUDGET
#   - reward tool_fast: correct * (1 + 0.1*15/(15+calls) + 0.05*clip((300-agent_s)/300,-1,1) - 0.1*token ramp(300..3000))
#   - agent time limit 600 s (10 min), training and eval; turn cap 17
#   - grader v3.1 on all 1,000 train tasks (prepared_v32); proxy tool-output cap 4,000 chars (as p4/p5)
#   - val118 (pinned grader, ONE rollout/task, eval concurrency EVAL_CONC) at u0 and every 50 updates
# Never kills a vLLM it did not start: if :8000 serves another model, it waits until the port is free.
# Run inside tmux. Stop: touch runs/p6_train1000f/STOP. Resume: rerun this script.
set -euo pipefail
export PIRL_THINKING=1   # p7: native <think> (reason, then act)
DA="$(cd "$(dirname "$0")" && pwd)"
SMOL="$(cd "$DA/../SmolDataEnvs" && pwd)"
run=p7_think_bash
kv=${KV:-fp8}
budget=${BUDGET:-32768}
eval_conc=${EVAL_CONC:-16}
total=400
every=50
mkdir -p "$DA/runs/$run"
exec >> "$DA/runs/p7_pipeline.log" 2>&1
say() { echo "$(date -u '+%F %T UTC') $*"; }
stop_requested() { [ -f "$DA/runs/$run/STOP" ]; }
harbor_ok() {  # healthy AND started with the v3.2 (1000-task v3.1 grader) train split + tool-output cap
  local pid
  pid=$(ss -ltnp | grep -E ":8201 " | grep -oE "pid=[0-9]+" | head -1 | cut -d= -f2 || true)
  [ -n "$pid" ] && curl -sf --max-time 5 localhost:8201/health >/dev/null &&
    tr '\0' '\n' < "/proc/$pid/environ" | grep -q "prepared_v32/datasets/train" &&
    tr '\0' '\n' < "/proc/$pid/environ" | grep -q "^HARBOR_TOOL_OUTPUT_CAP=4000$" &&
    tr '\0' '\n' < "/proc/$pid/environ" | grep -q "^HARBOR_PI_TOOLS=bash$"
}
ours_on_8000() { [ "$(cat "$DA/runs/vllm.mode" 2>/dev/null)" = "chain1_seqs64_kv${kv}_THINKING" ] &&
  curl -sf --max-time 5 localhost:8000/v1/models | grep -q "LiquidAI/LFM2.5-2.6B"; }
wait_for_gpu() {  # someone else's server on :8000 -> wait, never kill it
  while ss -ltn | grep -qE ":8000 " && ! ours_on_8000; do
    stop_requested && return 2
    say "port 8000 is serving another model; waiting for it to be stopped"
    sleep 300
  done
}
ensure_services() {
  wait_for_gpu || return 2
  if ! harbor_ok; then
    pid=$(ss -ltnp | grep -E ":8201 " | grep -oE "pid=[0-9]+" | head -1 | cut -d= -f2 || true)
    if [ -n "$pid" ]; then kill "$pid"; for _ in $(seq 30); do ss -ltn | grep -qE ":8201 " || break; sleep 1; done; fi
    say "starting Harbor: v3.1-graded 1000-task train split (prepared_v32), tool-output cap 4000"
    OPENENV_DATASETS="$DA/prepared/datasets/eval,$SMOL/prepared/datasets/test,$DA/prepared_v32/datasets/train" \
      HARBOR_TOOL_OUTPUT_CAP=4000 HARBOR_PI_TOOLS=bash TRL_EXPERIMENTAL_SILENCE=1 setsid nohup "$SMOL/.venv/bin/python" -u \
      "$DA/serve_harbor.py" --port 8201 --capture-port 8101 --trials "$SMOL/runs/trials_data_agent" \
      >> "$DA/runs/harbor_p7.log" 2>&1 < /dev/null &
    for _ in $(seq 60); do harbor_ok && break; sleep 2; done
    harbor_ok
  fi
  if ! ours_on_8000 || ! curl -sf --max-time 5 localhost:8000/health >/dev/null; then
    { CHAIN=1 SEQS=64 KV="$kv" "$DA/restart_vllm.sh" && echo "chain1_seqs64_kv${kv}_THINKING" > "$DA/runs/vllm.mode"; }
  fi
  if curl -sf localhost:8000/is_sleeping | grep -q true; then
    if ! curl -sf -X POST localhost:8000/wake_up >/dev/null; then
      say "wake failed; restarting vLLM"
      { CHAIN=1 SEQS=64 KV="$kv" "$DA/restart_vllm.sh" && echo "chain1_seqs64_kv${kv}_THINKING" > "$DA/runs/vllm.mode"; }
    fi
  fi
}
train_to() {
  local target=$1
  for attempt in 1 2 3 4; do
    stop_requested && { say "STOP requested"; return 2; }
    ensure_services || { say "STOP requested"; return 2; }
    say "training to update $target (attempt $attempt)"
    if TASKS=train798 CHAIN=1 CONC=32 TIMEOUT=600 "$DA/train_grpo.sh" "$run" "$target" \
      --sampler sequential --groups 4 --num-generations 8 --hard-generations 8 \
      --step-limit 25 --lora-r 16 --lora-alpha 32 --reward tool_fast --std-floor 0.1 --lr 1e-5 \
      --fp8-base --fp8-mode vllm --is-correction tis --tis-cap 2 --token-budget "$budget" \
      --save-steps "$every" --max-infra-fail 0.1 --data-root "$DA/prepared_v32" >> "$DA/runs/$run.log" 2>&1; then
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
  local result="$DA/results/val118_${name}_r1.json"
  if [ -f "$result" ] && "$SMOL/.venv/bin/python" -c \
    'import json,sys; r=json.load(open(sys.argv[1])); sys.exit(r.get("graded") != 118)' "$result"; then
    say "evaluation $name already complete"; return 0
  fi
  stop_requested && return 2
  ensure_services || return 2
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
  say "val118 at update $step ($model), one rollout/task (118 trials), concurrency $eval_conc"
  (cd "$SMOL"; PIRL_PREFIX_CHAIN=1 TRL_EXPERIMENTAL_SILENCE=1 .venv/bin/python -u pirl/evaluate.py \
    --label "$name" --model "$model" --split eval --data "$DA/prepared" --task-file "$DA/manifests/val118.json" \
    --rollouts-per-task 1 --step-limit 25 --concurrency "$eval_conc" --agent-timeout 600 --server http://127.0.0.1:8201 \
    --vllm-url http://127.0.0.1:8000 --trials "$SMOL/runs/trials_data_agent" \
    --output "$DA/runs/val118_${name}_r1" --results-json "$result") \
    >> "$DA/runs/val118_${name}_r1.log" 2>&1
  "$SMOL/.venv/bin/python" - "$name" "$result" "$step" <<'PY'
import httpx,json,sys
r=json.load(open(sys.argv[2]));print('validation',sys.argv[1],'graded',r['graded'],'pass@1',r['pass_at_1'])
if sys.argv[3] != '0':
    httpx.post('http://127.0.0.1:8000/v1/unload_lora_adapter',json={'lora_name':sys.argv[1]},timeout=120).raise_for_status()
if r['graded'] != 118: raise SystemExit('incomplete validation; rerun supervisor to retry missing grades')
PY
}
say "starting $run: $total updates (2 passes x 798 learnable tasks, 4x8), thinking ON, pi bash-only, turn cap 25, LoRA r16, reward tool_fast, 10-min cap"
evaluate_at 0 || exit 0
for target in $(seq $every $every $total); do
  train_to "$target" || exit 0
  evaluate_at "$target" || exit 0
done
say "completed $total updates and all scheduled validations"
touch "$DA/runs/$run/DONE"
