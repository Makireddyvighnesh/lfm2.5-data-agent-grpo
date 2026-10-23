"""Run the SmolDataEnvs Harbor server (task API + capture proxy) for local Docker sandboxes.

Same app as FineEnvs' envs/harbor server, with two local-only settings:
- the capture proxy is advertised at the host's primary IP, which every sandbox network can route to
  (OpenEnv's `direct` exposure otherwise advertises 127.0.0.1, which inside a container is itself);
- concurrency and the per-call output cap come from the command line.
No OpenEnv source is modified: the `direct` forwarder is re-registered with a different host.
"""

import argparse
import os
import subprocess
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pirl.common import DATA, MAX_TOKENS


def host_address():
    """The host's primary IP, which every sandbox network can route to.

    Not the docker0 gateway: with custom default-address-pools, sandboxes live on separate bridges
    and Docker's isolation rules drop their traffic to another bridge's IP (172.17.0.1 timed out).
    """
    out = subprocess.check_output(["ip", "-4", "route", "get", "1.1.1.1"], text=True).split()
    return out[out.index("src") + 1]


def pi_session_as_atif(trial_dir):
    """OpenEnv's pi reader, keeping each assistant message's text and tool-call flag.

    Upstream keeps only `completion_tokens`, so `reconcile` cannot recognise the capture proxy's
    zero-token "Step budget exhausted; stopping." reply (it matches on `message`) and every pi
    rollout that reaches the context limit becomes a FATAL turn_mismatch.
    """
    import json

    sessions = sorted((Path(trial_dir) / "agent" / "pi" / "sessions").glob("*.jsonl"))
    if not sessions:
        return None
    steps = []
    for raw in sessions[-1].read_text().splitlines():
        try:
            record = json.loads(raw)
        except ValueError:
            continue
        message = record.get("message") or {}
        if record.get("type") != "message" or message.get("role") != "assistant":
            continue
        content = message.get("content") or []
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        text = "".join(c.get("text", "") for c in content if c.get("type") == "text")
        calls = [c for c in content if c.get("type") == "toolCall"]
        usage = message.get("usage") or {}
        steps.append(
            {
                "source": "agent",
                "message": text,
                "tool_calls": calls or None,
                "metrics": {
                    "completion_tokens": int(usage.get("output") or 0),
                    "prompt_tokens": int(usage.get("input") or 0),
                },
            }
        )
    return {"steps": steps} if steps else None


def install_pi_trace_fix():
    from openenv.harbor import atif

    pi_session_as_atif.__name__ = "_pi_session_as_atif"  # keeps trace_source == "pi_session"
    atif._FALLBACK_TRACES = (pi_session_as_atif,)


PI_VERSION = "1.0.0"


def install_pi_prebaked_skip():
    """Skip Harbor's per-rollout nvm + npm pi install when the sandbox image already has it.

    sandbox_image/Dockerfile bakes Node 22 + pi into the task base image; the install otherwise
    costs ~22 s of every ~45 s rollout. Falls back to the normal install if pi is missing.
    """
    from harbor.agents.installed.pi import Pi

    original = Pi.install

    async def install(self, environment):
        check = await environment.exec(command=". ~/.nvm/nvm.sh >/dev/null 2>&1; pi --version")
        if check.return_code == 0 and (check.stdout or "").strip().endswith(PI_VERSION):
            self._version = PI_VERSION
            return
        await original(self, environment)

    Pi.install = install


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8200)
    p.add_argument("--capture-port", type=int, default=8100)
    p.add_argument("--max-envs", type=int, default=128)  # >= 2x rollout concurrency: closed sessions free slots late
    p.add_argument("--trials", default=str(DATA.parent / "runs" / "trials"))
    a = p.parse_args()

    # Docker here is a snap with a private /tmp: compose files Harbor writes to /tmp are invisible to it.
    tmp = DATA.parent / "runs" / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmp)
    import tempfile

    tempfile.tempdir = str(tmp)
    os.environ["SMOLDATA_DATA"] = str(DATA)
    os.environ["OPENENV_EXPOSE"] = "direct"
    os.environ["OPENENV_CAPTURE_PORT"] = str(a.capture_port)
    os.environ["MAX_CONCURRENT_ENVS"] = str(a.max_envs)
    os.environ["OPENENV_MAX_OUTPUT_TOKENS"] = str(MAX_TOKENS)
    os.environ["OPENENV_HARBOR_TRIALS_DIR"] = a.trials
    os.environ.setdefault("OPENENV_HARBOR_REWARD_KEY", "correctness,reward")
    # The tutorial pins harness versions for reproducibility; pin pi too once a version is chosen.
    os.environ.setdefault("OPENENV_HARBOR_AGENT_VERSIONS", os.environ.get("PIRL_AGENT_VERSIONS", "{}"))
    token = Path.home() / ".cache/huggingface/token"
    if "HF_TOKEN" not in os.environ and token.exists():
        os.environ["HF_TOKEN"] = token.read_text().strip()  # sandboxes pull task data from an HF bucket

    from openenv.core.harness.capture import forwarding

    forwarding._FORWARDERS["direct"] = partial(forwarding.DirectExposure, advertise_host=host_address())
    install_pi_trace_fix()
    install_pi_prebaked_skip()

    import uvicorn
    from smoldataenv_harbor.server import app

    uvicorn.run(app, host="127.0.0.1", port=a.port, ws_ping_timeout=1800, log_level="warning")


if __name__ == "__main__":
    main()
