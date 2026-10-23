#!/usr/bin/env bash
# Keep a GRPO run going until it reaches its step target. On any trainer exit it checks/restarts the
# Harbor server and vLLM, then relaunches pirl/grpo.py, which resumes from <output>/state_latest.
#   usage: pirl/run_resumable.sh <output_dir> [grpo.py args...]     (logs: <output_dir>.log)
#   stop: python3 pirl/stop_run.py <output_dir>, or touch <output_dir>/STOP
set -u
cd "$(dirname "$0")/.."
out=$1; shift
mkdir -p "$out"
echo $$ > "$out/supervisor.pid"
log="$out.log"
source .venv/bin/activate
export TRL_EXPERIMENTAL_SILENCE=1
steps=$(python - "$@" <<'EOF'
import sys
a = sys.argv[1:]
print(a[a.index("--steps") + 1] if "--steps" in a else 200)
EOF
)
done_steps() { python - "$out" <<'PY'
import json, sys
from pathlib import Path
from pirl.grpo import recover_state
path = Path(sys.argv[1]) / "state_latest"
print(json.loads((path / "state.json").read_text())["step"] if recover_state(path) else 0)
PY
}

say() { echo "$(date -u +%FT%T) $*" | tee -a "$log"; }   # supervisor events go to the tmux pane too
restarts=0
trap 'if [ -n "${tpid:-}" ]; then kill -TERM "$tpid" 2>/dev/null; fi; if [ -n "${mirror:-}" ]; then kill -TERM -- "-$mirror" 2>/dev/null; fi; exit 0' TERM INT
while [ "$(done_steps)" -lt "$steps" ]; do
  [ -f "$out/STOP" ] && { say "STOP file found"; break; }
  curl -sf localhost:8200/health >/dev/null || { say "harbor down, restarting"; pirl/svc.sh stop harbor; pirl/svc.sh start harbor 2>&1 | tee -a "$log"; }
  # A vLLM that is alive but still loading is not "down": give it up to 5 minutes before replacing it.
  if ! curl -sf localhost:8000/health >/dev/null; then
    vpid=$(cat runs/vllm.pid 2>/dev/null)
    if [ -n "$vpid" ] && kill -0 "$vpid" 2>/dev/null; then
      say "vLLM (pid $vpid) not healthy yet, waiting"
      for _ in $(seq 60); do curl -sf localhost:8000/health >/dev/null && break; sleep 5; done
    fi
    if ! curl -sf localhost:8000/health >/dev/null; then
      say "vLLM down, (re)starting (~1.5 min)"
      pirl/svc.sh stop vllm
      pirl/svc.sh start vllm --output runs/vllm_train --prefix-caching --sleep-mode --gpu-mem 0.62 2>&1 | tee -a "$log"
    fi
  fi
  say "vLLM healthy"
  # Another process holding GPU memory (a leftover trainer, a test) makes the trainer OOM at load.
  for _ in $(seq 60); do
    others=$(nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader | grep -v -i vllm)
    [ -z "$others" ] && break
    say "waiting for GPU: $others"; sleep 10
  done
  say "launch trainer (restart #$restarts) at step $(done_steps); per-update lines follow (full log: $log)"
  python -u pirl/grpo.py --output "$out" "$@" >> "$log" 2>&1 &
  echo $! > "$out/trainer.pid"
  tpid=$!
  # Mirror one line per finished update to the pane, so the session never looks frozen.
  setsid bash -c 'tail -n0 -F "$1" 2>/dev/null | grep --line-buffered -E "^UPDATE |^ROLLOUT |^STOP |resumed at|Traceback|Error:"' _ "$log" &
  mirror=$!
  while kill -0 "$tpid" 2>/dev/null; do
    if [ -f "$out/STOP" ]; then
      say "STOP requested; terminating trainer at the last saved checkpoint"
      kill -TERM "$tpid" 2>/dev/null
      break
    fi
    sleep 1
  done
  wait $tpid
  code=$?
  kill -TERM -- "-$mirror" 2>/dev/null
  say "trainer exited code=$code at step $(done_steps)"
  [ "$(done_steps)" -ge "$steps" ] && break
  [ -f "$out/STOP" ] && break
  restarts=$((restarts + 1))
  [ $restarts -gt "${PIRL_MAX_RESTARTS:-3}" ] && { say "restart limit reached"; exit 1; }
  # Harbor owns trial teardown. Never delete all oe-* containers: other runs may own them.
  sleep 30
done
say "finished at step $(done_steps)"
