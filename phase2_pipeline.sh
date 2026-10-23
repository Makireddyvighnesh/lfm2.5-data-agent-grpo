#!/usr/bin/env bash
# Phase 2, unattended (run inside tmux session "dataagent-p2"). Steps, each skipped if already done:
#   1. start point: r2-final on val118 under the prefix-chain template (FP8, 16k, T=1)
#   2. arm A (blog tool reward)   20 updates on hard50, then val118
#   3. arm B (r2 context reward)  20 updates on hard50, then val118
#   4. arm C (correctness only)   20 updates on hard50, then val118
#   5. r2-final on val118 under the native template (template check of the start point)
#   6. results/phase2_summary.md
# A crashed arm is resumed from its state_latest (up to 4 attempts); services are restarted when unhealthy.
# Log: runs/phase2_pipeline.log   Re-run this script to resume after any interruption.
set -uo pipefail
DA="$(cd "$(dirname "$0")" && pwd)"
SMOL="$(cd "$DA/../SmolDataEnvs" && pwd)"
LOG="$DA/runs/phase2_pipeline.log"
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

ensure_harbor() {
  curl -sf localhost:8201/health >/dev/null && return 0
  say "starting Harbor (:8201)"
  ( cd "$SMOL" && source .venv/bin/activate && export TRL_EXPERIMENTAL_SILENCE=1 && \
    OPENENV_DATASETS="$DA/prepared/datasets/eval,$SMOL/prepared/datasets/test,$SMOL/prepared/datasets/train" \
    setsid nohup python "$DA/serve_harbor.py" --port 8201 --capture-port 8101 --trials "$SMOL/runs/trials_data_agent" \
      > "$DA/runs/harbor.log" 2>&1 < /dev/null & )
  for _ in $(seq 60); do curl -sf localhost:8201/health >/dev/null && { say "Harbor up"; return 0; }; sleep 2; done
  say "Harbor failed to start"; return 1
}

ensure_vllm() {  # $1 = 1 (prefix-chain) or 0 (native)
  local want="chain$1_seqs64"
  if curl -sf localhost:8000/health >/dev/null && [ "$(cat "$DA/runs/vllm.mode" 2>/dev/null)" = "$want" ]; then
    curl -s localhost:8000/is_sleeping | grep -q true && curl -s -X POST localhost:8000/wake_up >/dev/null
    return 0
  fi
  say "restarting vLLM ($want)"
  if CHAIN=$1 SEQS=64 "$DA/restart_vllm.sh" >> "$LOG" 2>&1; then echo "$want" > "$DA/runs/vllm.mode"; return 0; fi
  say "vLLM failed to start"; return 1
}

clean_sandboxes() { docker ps -q --filter "name=^oe-" | xargs -r docker rm -f > /dev/null 2>&1 || true; }

arm_done() {
  python3 -c "import json,sys; r=json.load(open('$DA/runs/$1/results.json')); sys.exit(0 if r.get('completed_steps',0)>=20 else 1)" 2>/dev/null
}

run_arm() {  # $1 = A|B|C  $2 = run name
  for attempt in 1 2 3 4; do
    arm_done "$2" && { say "arm $1 ($2): complete"; return 0; }
    ensure_harbor && ensure_vllm 1 || { sleep 60; continue; }
    clean_sandboxes
    say "arm $1 ($2): training, attempt $attempt"
    "$DA/run_p2_arm.sh" "$1" >> "$DA/runs/$2.log" 2>&1
    say "arm $1 ($2): trainer exited with code $?"
  done
  arm_done "$2" || say "arm $1 ($2): NOT complete after 4 attempts; moving on"
}

evaluate() {  # $1 = adapter dir  $2 = name  $3 = chain 1|0
  if [ -f "$DA/results/val118_$2.json" ]; then say "eval $2: already done"; return 0; fi
  [ -d "$1" ] || { say "eval $2: adapter $1 missing; skipped"; return 1; }
  ensure_harbor && ensure_vllm "$3" || return 1
  clean_sandboxes
  say "eval $2: val118 (chain=$3)"
  CHAIN=$3 "$DA/eval_checkpoint.sh" "$1" "$2" val118 >> "$LOG" 2>&1 || say "eval $2: FAILED (see log)"
  [ -f "$DA/results/val118_$2.json" ] && say "eval $2: $(python3 -c "import json; s=json.load(open('$DA/results/val118_$2.json')); print(f\"pass@1 {s['pass_at_1']:.3f}, out-of-ctx {s['context_exhausted_rate']:.2f}, tool calls {s['mean_tool_calls']:.1f}\")")"
}

say "=== phase 2 pipeline start (pid $$)"
evaluate "$DA/checkpoints/r2-final" r2-final-chain 1
for spec in A:p2_A_tool B:p2_B_context C:p2_C_correctness; do
  arm=${spec%%:*}; run=${spec##*:}
  run_arm "$arm" "$run"
  evaluate "$DA/runs/$run/state_latest/adapter" "$run-final" 1
done
evaluate "$DA/checkpoints/r2-final" r2-final-native 0
python3 "$DA/analysis/phase2_summary.py" >> "$LOG" 2>&1 && say "summary written: results/phase2_summary.md"
say "=== phase 2 pipeline finished"
