#!/usr/bin/env bash
# Start/stop the long-running services by PID file (pkill -f would match the calling shell).
# usage: pirl/svc.sh start vllm <serve.py args...> | stop vllm | start harbor <args> | stop harbor
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
export TRL_EXPERIMENTAL_SILENCE=1
action=$1; name=$2; shift 2
pidf=runs/$name.pid
case "$action" in
  stop)
    if [ -f "$pidf" ]; then kill "$(cat "$pidf")" 2>/dev/null; fi
    for _ in $(seq 60); do [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null || break; sleep 1; done
    rm -f "$pidf" ;;
  start)
    case "$name" in
      vllm) script=pirl/serve.py; url=localhost:8000/health ;;
      harbor) script=pirl/harbor_server.py; url=localhost:8200/health ;;
    esac
    setsid nohup python "$script" "$@" > "runs/$name.log" 2>&1 &
    echo $! > "$pidf"
    for _ in $(seq 200); do curl -sf "$url" >/dev/null && { echo "$name up"; exit 0; }; kill -0 "$(cat "$pidf")" 2>/dev/null || { echo "$name died"; tail -20 "runs/$name.log"; exit 1; }; sleep 3; done
    echo "$name timeout"; exit 1 ;;
esac
