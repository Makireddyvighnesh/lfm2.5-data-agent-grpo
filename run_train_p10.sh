#!/usr/bin/env bash
# p10_mh_r32: p8 multi-harness recipe trained FROM SCRATCH (fresh LoRA on base LFM2.5-2.6B, no init adapter) on all
# 1000 train tasks (incl. the 202 p6 never solved) with DAPO dynamic sampling (groups all correct / all wrong are
# dropped and replaced; each update trains 8 groups with correctness contrast).
# Changes vs p8: no --init-adapter, LoRA r32 / alpha 64, 8 tasks x 8 rollouts per update (64 rollouts; packed
# 32k-token micro-batches = gradient accumulation into one optimizer step), 200 updates,
# cosine over all 200 (--schedule-steps), checkpoints every 50, and val118 + test250 (r1, 24k) on pi /
# mini-swe-agent / codex every 100 updates (trainer pauses, eval runs alone on the GPU, trainer resumes).
# Thinking OFF (p8: ON). Same as p8: harnesses pi,mini-swe-agent,codex (task index mod 3), prefix chain, pi bash-only, turn cap
# 25, reward tool_ctx, lr 3e-5 cosine (warmup 10, min 3e-6), FP8 + TIS cap 2.
# Speed (from u42, 2026-10-27): --async-groups 14 keeps 14 unfinished groups generating (conc 112, vLLM 96 seqs,
# 6 LoRA slots, LoRA rank 32 = adapter rank: rank 64 padded every LoRA matmul); --dynamic-skip removes a task once a visit was all correct and skips 50% of revisits after an all-wrong visit (from u53); agent
# wall-clock cap 900 s because in-flight rollouts now pause ~4.5 min per training phase (active cap stays ~600 s;
# evals keep 600 s). u1-41 ran with 1 extra group in flight, conc 72, 64 seqs, 600 s.
# usage: ./run_train_p10.sh   (run in tmux; resumable: re-run to continue; stop: touch runs/p10_mh_r32/STOP)
#        EPOCH_STOP=1000 ./run_train_p10.sh   train until every task position < 1000 is done (end of epoch 1), keep
#        that adapter as runs/p10_mh_r32/epoch1, run test250 (r1) on EVAL_HS (default: the 3 training harnesses +
#        opencode + claude-code), then exit. EPOCH_STOP=2000 = epoch 2 (runs/p10_mh_r32/epoch2, label p10e2), etc.
set -uo pipefail
DA="$(cd "$(dirname "$0")" && pwd)"; SMOL="$(cd "$DA/../SmolDataEnvs" && pwd)"
run=p10_mh_r32; total=200; every=100; kv=fp8; mode=chain1_seqs96_kvfp8_NOTHINK_len24k_lr32
HS=(pi mini-swe-agent codex)
mkdir -p "$DA/runs/$run"
exec >> "$DA/runs/${run}_pipeline.log" 2>&1
say() { echo "$(date -u '+%F %T UTC') $*"; }
export PIRL_THINKING=0  # thinking OFF (user decision 2026-10-26); vLLM template follows it

harbor_ok() {  # healthy AND started with the v3.2 train split + tool-output cap + pi bash-only (p7/p8 Harbor)
  local pid
  pid=$(ss -ltnp | grep -E ":8201 " | grep -oE "pid=[0-9]+" | head -1 | cut -d= -f2 || true)
  [ -n "$pid" ] && curl -sf --max-time 5 localhost:8201/health >/dev/null &&
    tr '\0' '\n' < "/proc/$pid/environ" | grep -q "prepared_v32/datasets/train" &&
    tr '\0' '\n' < "/proc/$pid/environ" | grep -q "^HARBOR_TOOL_OUTPUT_CAP=4000$" &&
    tr '\0' '\n' < "/proc/$pid/environ" | grep -q "^HARBOR_PI_TOOLS=bash$" &&
    tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q -- "--max-envs 256"  # >= 2x the 112 rollout slots
}
ensure_services() {  # FRESH_HARBOR=1: restart Harbor even if healthy (sessions of abandoned rollouts are never freed)
  if [ "${FRESH_HARBOR:-0}" = 1 ] || ! harbor_ok; then
    pid=$(ss -ltnp | grep -E ":8201 " | grep -oE "pid=[0-9]+" | head -1 | cut -d= -f2 || true)
    if [ -n "$pid" ]; then kill "$pid"; for _ in $(seq 120); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done; kill -9 "$pid" 2>/dev/null || true; fi
    for _ in $(seq 60); do ss -ltn | grep -qE ":(8201|8101) " || break; sleep 1; done  # capture port frees last
    say "starting Harbor :8201 (eval + test + prepared_v32 train, tool-output cap 4000, pi bash-only)"
    OPENENV_DATASETS="$DA/prepared/datasets/eval,$SMOL/prepared/datasets/test,$DA/prepared_v32/datasets/train" \
      HARBOR_TOOL_OUTPUT_CAP=4000 HARBOR_PI_TOOLS=bash TRL_EXPERIMENTAL_SILENCE=1 setsid nohup "$SMOL/.venv/bin/python" -u \
      "$DA/serve_harbor.py" --port 8201 --capture-port 8101 --max-envs 256 --trials "$SMOL/runs/trials_data_agent" \
      >> "$DA/runs/harbor_p10.log" 2>&1 < /dev/null &
    for _ in $(seq 60); do harbor_ok && break; sleep 2; done
    harbor_ok || { say "Harbor failed to start"; return 1; }
  fi
  if [ "$(cat "$DA/runs/vllm.mode" 2>/dev/null)" != "$mode" ] || ! curl -sf --max-time 5 localhost:8000/health >/dev/null; then
    say "starting vLLM (24k, FP8, prefix chain, thinking OFF)"
    MAX_LEN=24576 CHAIN=1 SEQS=96 KV=$kv LORA_RANK=32 MAX_LORAS=6 "$DA/restart_vllm.sh" && echo "$mode" > "$DA/runs/vllm.mode" || return 1
  fi
  if curl -sf localhost:8000/is_sleeping | grep -q true; then curl -sf -X POST localhost:8000/wake_up >/dev/null; fi
  return 0
}

train_to() {  # resumes from runs/$run/state_latest; fresh LoRA on the first start; extra args go to grpo.py
  local target=$1; shift
  TASKS=train1000 CHAIN=1 CONC=112 TIMEOUT=900 "$DA/train_grpo.sh" "$run" "$target" "$@" \
    --harnesses pi,mini-swe-agent,codex --sampler sequential --groups 8 --num-generations 8 --hard-generations 8 --dynamic-sampling --dynamic-skip --skip-wrong-frac 0.5 --async-groups 14 \
    --step-limit 25 --lora-r 32 --lora-alpha 64 --reward tool_ctx --std-floor 0.1 \
    --lr 3e-5 --lr-schedule cosine --warmup-steps 10 --min-lr 3e-6 --schedule-steps "$total" --stream-groups 1 \
    --fp8-base --fp8-mode vllm --is-correction tis --tis-cap 2 --token-budget 32768 \
    --save-steps 50 --allow-change agent_timeout --max-infra-fail 0.1 --data-root "$DA/prepared_v32" >> "$DA/runs/$run.log" 2>&1
}

summ() { "$SMOL/.venv/bin/python" -c 'import json,sys; s=json.load(open(sys.argv[1])); print(sys.argv[2], "graded", s["graded"], "pass@1", round(s["pass_at_1"],3), "ctx_out", round(s.get("context_exhausted_rate") or 0,3), "tools", round(s.get("mean_tool_calls") or 0,1))' "$1" "$2"; }

eval_ckpt() {  # val118 + test250, r1, each training harness
  local u=$1 name="p10u$1" adapter="$DA/runs/$run/checkpoint-$1/adapter"
  ensure_services || return 1
  "$SMOL/.venv/bin/python" -c 'import httpx,sys; print(httpx.post("http://127.0.0.1:8000/v1/load_lora_adapter",json={"lora_name":sys.argv[1],"lora_path":sys.argv[2]},timeout=120).text[:120])' "$name" "$adapter"
  for h in "${HS[@]}"; do
    res="$DA/results/val118_mh24k_${h}_${name}_r1.json"
    if [ ! -f "$res" ]; then
      say "val118 $h $name"
      (cd "$SMOL"; PIRL_PREFIX_CHAIN=1 TRL_EXPERIMENTAL_SILENCE=1 .venv/bin/python -u pirl/evaluate.py \
        --label "mh24k_${h}_$name" --model "$name" --harness "$h" --split eval --data "$DA/prepared" --task-file "$DA/manifests/val118.json" \
        --rollouts-per-task 1 --step-limit 25 --concurrency 32 --agent-timeout 600 \
        --server http://127.0.0.1:8201 --vllm-url http://127.0.0.1:8000 --trials "$SMOL/runs/trials_data_agent" \
        --output "$DA/runs/val118_mh24k_${h}_${name}_r1" --results-json "$res") > "$DA/runs/val118_mh24k_${h}_${name}_r1.log" 2>&1
    fi
    [ -f "$res" ] && say "$(summ "$res" "val118 $h $name")"
  done
  test250_all "$name" "$adapter"
}

test250_all() {  # test250 r1 on each harness in $3 (default: the training harnesses) for one adapter
  local name=$1 adapter=$2 hs=${3:-${HS[*]}} h res
  ensure_services || return 1
  for h in $hs; do
    res="$DA/results/test250_mh24k_${h}_${name}_r1.json"
    [ -f "$res" ] || { say "test250 $h $name"; HARNESS=$h "$DA/run_test250_p7.sh" "mh24k_${h}_${name}" "$name" "$adapter"; }
    [ -f "$res" ] && say "$(summ "$res" "test250 $h $name")"
  done
}

if [ -n "${EPOCH_STOP:-}" ]; then
  n=$(( EPOCH_STOP / 1000 )); ep="$DA/runs/$run/epoch$n"
  if [ ! -f "$ep/adapter/adapter_model.safetensors" ]; then
    ensure_services || exit 1
    say "training $run until task position $EPOCH_STOP is done (end of epoch $n)"
    FRESH_HARBOR=1 ensure_services || exit 1
    train_to "$total" --stop-at-pos "$EPOCH_STOP"; rc=$?
    grep -q "all task positions < $EPOCH_STOP are done" "$DA/runs/$run.log" || { say "trainer exited $rc before the end of epoch $n"; exit 1; }
    mkdir -p "$ep" && cp -r "$DA/runs/$run/state_latest/adapter" "$ep/adapter" && cp "$DA/runs/$run/state_latest/state.json" "$ep/state.json"
    say "epoch $n done at update $(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["step"])' "$ep/state.json"); adapter kept in $ep"
  fi
  test250_all "p10e$n" "$ep/adapter" "${EVAL_HS:-pi mini-swe-agent codex opencode claude-code}" || { say "epoch-$n test250 failed"; exit 1; }
  say "epoch-$n eval done"
  exit 0
fi

for target in $(seq "$every" "$every" "$total"); do
  [ -f "$DA/runs/$run/STOP" ] && { say "STOP present; exiting"; exit 0; }
  if [ ! -f "$DA/runs/$run/checkpoint-$target/adapter/adapter_model.safetensors" ]; then
    ensure_services || exit 1
    FRESH_HARBOR=1 ensure_services || exit 1
    say "training $run to update $target / $total"
    train_to "$target"; rc=$?
    [ -f "$DA/runs/$run/checkpoint-$target/adapter/adapter_model.safetensors" ] || { say "trainer exited $rc before checkpoint-$target"; exit 1; }
  fi
  eval_ckpt "$target" || { say "eval of checkpoint-$target failed"; exit 1; }
done
say "$run done"
