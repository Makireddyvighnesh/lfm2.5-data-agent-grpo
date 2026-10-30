#!/usr/bin/env bash
# p12_mh4_r32 (2026-10-29): fresh start of the p11 design with every fix active from update 1.
# Fresh LoRA r32 on base LFM2.5-2.6B, thinking OFF, 1000 train tasks, 8 tasks x 8 rollouts, 24k, lr 3e-5 cosine/200,
# FP8 + TIS cap 2, async generation (14 groups), DAPO dynamic sampling + skip rules (solved removed, 50% of
# all-wrong revisits skipped).
# Training harnesses: pi, mini-swe-agent, codex, prime-agent (task index mod 4); OpenCode and Claude Code HELD OUT.
# Reward tool_ctx_err: tool_ctx - 0.1 per call to a tool not offered in that request (cap 0.3), read from the rollout
# result (the p11 counter read the training trace and saw 0). Groups are also kept when they differ in such calls.
# Speed: training turn cap 17 (blog value; evals keep 25), probe sampling (--probe-hard 4) for tasks last seen all wrong.
# usage: EPOCH_STOP=1000 ./run_train_p12.sh   train to the end of epoch 1 (runs/p12_mh4_r32/epoch1), then test250 r1
#        on opencode claude-code pi mini-swe-agent codex prime-agent (held-out first), label p12e1. EPOCH_STOP=2000 =
#        epoch 2, etc. Stop: touch runs/p12_mh4_r32/STOP. Resumable: re-run the same command.
# Epoch 2+ (2026-10-29, after the e1 held-out diagnosis): Pi gets `read` back (bash,read) so file reading is under RL
# pressure, and reward tool_ctx_read = tool_ctx_err + capped-output penalty (0.05 per tool output the proxy cut at
# 4000 chars, max 0.2) + own-context bonus graded 4k->12k instead of flat at 15k. Epoch 1 is unchanged.
# Epoch 3+ (2026-10-30, after the e2 review): reward tool_ctx_read2 (capped-output penalty on correct answers only; on
# wrong answers it ranked reading nothing above inspecting the data); all-correct groups whose rewards spread by
# >= 0.03 are kept; 25% of revisits to tasks solved 8/8 are run instead of skipped; cosine span and update cap
# 200 -> 300 (at u127 the lr was 1.2e-5 and fell to the 3e-6 floor by u200; epoch 2 needed 56 updates).
set -uo pipefail
total=200; EXTRA=()
if [ "${EPOCH_STOP:-0}" -ge 3000 ]; then
  PI_TOOLS=bash,read; REWARD=tool_ctx_read2; total=300; EXTRA=(--keep-reward-spread 0.03 --solved-revisit-frac 0.25)
elif [ "${EPOCH_STOP:-0}" -ge 2000 ]; then PI_TOOLS=bash,read; REWARD=tool_ctx_read
else PI_TOOLS=bash; REWARD=tool_ctx_err; fi
DA="$(cd "$(dirname "$0")" && pwd)"; SMOL="$(cd "$DA/../SmolDataEnvs" && pwd)"
run=p12_mh4_r32; every=100; kv=fp8; mode=chain1_seqs96_kvfp8_NOTHINK_len24k_lr32
HS=(pi mini-swe-agent codex prime-agent)
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
    tr '\0' '\n' < "/proc/$pid/environ" | grep -q "^HARBOR_PI_TOOLS=$PI_TOOLS$" &&
    tr '\0' '\n' < "/proc/$pid/environ" | grep -q "^PRIME_AGENT_CONTEXT=24576$" &&
    tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q -- "--max-envs 256"  # >= 2x the 112 rollout slots
}
ensure_services() {  # FRESH_HARBOR=1: restart Harbor even if healthy (sessions of abandoned rollouts are never freed)
  if [ "${FRESH_HARBOR:-0}" = 1 ] || ! harbor_ok; then
    pid=$(ss -ltnp | grep -E ":8201 " | grep -oE "pid=[0-9]+" | head -1 | cut -d= -f2 || true)
    if [ -n "$pid" ]; then kill "$pid"; for _ in $(seq 120); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done; kill -9 "$pid" 2>/dev/null || true; fi
    for _ in $(seq 60); do ss -ltn | grep -qE ":(8201|8101) " || break; sleep 1; done  # capture port frees last
    say "starting Harbor :8201 (eval + test + prepared_v32 train, tool-output cap 4000, pi tools $PI_TOOLS)"
    OPENENV_DATASETS="$DA/prepared/datasets/eval,$SMOL/prepared/datasets/test,$DA/prepared_v32/datasets/train" \
      HARBOR_TOOL_OUTPUT_CAP=4000 HARBOR_PI_TOOLS=$PI_TOOLS PRIME_AGENT_CONTEXT=24576 TRL_EXPERIMENTAL_SILENCE=1 setsid nohup "$SMOL/.venv/bin/python" -u \
      "$DA/serve_harbor.py" --port 8201 --capture-port 8101 --max-envs 256 --trials "$SMOL/runs/trials_data_agent" \
      >> "$DA/runs/harbor_p12.log" 2>&1 < /dev/null &
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
    --harnesses pi,mini-swe-agent,codex,prime-agent --sampler sequential --groups 8 --num-generations 8 --hard-generations 8 --dynamic-sampling --dynamic-keep-tool-contrast --probe-hard 4 --dynamic-skip --skip-wrong-frac 0.5 --async-groups 14 \
    --step-limit 17 --lora-r 32 --lora-alpha 64 --reward $REWARD --allow-change reward --std-floor 0.1 \
    --lr 3e-5 --lr-schedule cosine --warmup-steps 10 --min-lr 3e-6 --schedule-steps "$total" --stream-groups 1 \
    --fp8-base --fp8-mode vllm --is-correction tis --tis-cap 2 --token-budget 32768 \
    --save-steps 50 --max-infra-fail 0.1 --data-root "$DA/prepared_v32" "${EXTRA[@]}" >> "$DA/runs/$run.log" 2>&1
}

summ() { "$SMOL/.venv/bin/python" -c 'import json,sys; s=json.load(open(sys.argv[1])); print(sys.argv[2], "graded", s["graded"], "pass@1", round(s["pass_at_1"],3), "ctx_out", round(s.get("context_exhausted_rate") or 0,3), "tools", round(s.get("mean_tool_calls") or 0,1))' "$1" "$2"; }

eval_ckpt() {  # val118 + test250, r1, each training harness
  local u=$1 name="p12u$1" adapter="$DA/runs/$run/checkpoint-$1/adapter"
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
  test250_all "p12e$n" "$ep/adapter" "${EVAL_HS:-opencode claude-code pi mini-swe-agent codex prime-agent}" || { say "epoch-$n test250 failed"; exit 1; }
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
