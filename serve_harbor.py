"""Harbor task API + capture proxy for Data Agent evals: SmolDataEnvs' pirl/harbor_server.py plus

- a `prime-agent` seam (Harbor has no adapter; see prime_agent_harness.py), registered at runtime so
  OpenEnv and Harbor stay unmodified;
- OpenCode pinned to the tutorial's 1.18.31 and its per-rollout npm install skipped when the sandbox
  image already has it (sandbox_image/Dockerfile), like pi's prebaked skip.

Serve the splits named in OPENENV_DATASETS, e.g.
  OPENENV_DATASETS=<eval dir>,<test dir> python serve_harbor.py --port 8201 --capture-port 8101 --trials <dir>
"""

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "SmolDataEnvs")]

OPENCODE_VERSION = "1.18.31"


def register_prime_agent():
    from openenv.harbor import seams

    seams.SEAMS["prime-agent"] = seams.Seam(
        name="prime-agent",
        dialect="openai_chat",
        import_path="prime_agent_harness:InterceptPrimeAgent",
        model_fmt=seams.PI_PROVIDER + "/{model}",
        kwargs=seams._pi,  # same intercept_config payload as pi
        status="untested",
        notes="Prime Agent 0.10.0; pi-format models.json and session log. No native step limit.",
    )


def install_opencode_prebaked_skip():
    from harbor.agents.installed.opencode import OpenCode

    original = OpenCode.install

    async def install(self, environment):
        check = await environment.exec(command=". ~/.nvm/nvm.sh >/dev/null 2>&1; opencode --version")
        if check.return_code == 0 and (check.stdout or "").strip().endswith(OPENCODE_VERSION):
            self._version = OPENCODE_VERSION
            return
        await original(self, environment)

    OpenCode.install = install


def install_mini_swe_prebaked_skip():
    """Skip mini-swe-agent's per-rollout `uv tool install` when the sandbox image already has it.

    Harbor reinstalls on every rollout, unpinned, and current litellm imports `mcp` without declaring it, so a
    fresh install fails at startup (ModuleNotFoundError: mcp). sandbox_image/Dockerfile.mh installs it with
    `--with mcp`; this keeps that install instead of replacing it.
    """
    from harbor.agents.installed.mini_swe_agent import MiniSweAgent

    original = MiniSweAgent.install

    async def install(self, environment):
        check = await environment.exec(command='export PATH="$HOME/.local/bin:$PATH"; mini-swe-agent --help >/dev/null 2>&1')
        if check.return_code == 0:
            return
        await original(self, environment)

    MiniSweAgent.install = install


def install_fast_teardown():
    """Kill sandboxes instead of stopping them gracefully.

    Harbor's main container runs `sh -c "sleep infinity"` as PID 1 with no init, which ignores SIGTERM, so every
    `docker compose down` waited Docker's full 10 s grace period (measured: kill->die median 10.2 s; teardown
    19 s per rollout, which the rollout waits for). Logs are bind-mounted and collected before stop, so there is
    nothing to flush: SIGKILL first, then Harbor's normal `down` removes the stack in about 1 s.
    """
    from harbor.environments.docker.docker import DockerEnvironment

    original = DockerEnvironment.stop

    async def stop(self, delete):
        if not delete or getattr(self, "_keep_containers", False):
            return await original(self, delete)
        await self.prepare_logs_for_host()  # chown the bind-mounted logs while the container is still alive
        try:
            await self._run_docker_compose_command(["kill"])
        except Exception:
            pass  # never block teardown; `down` below still removes everything

        async def _already_done():
            return None

        self.prepare_logs_for_host = _already_done  # instance override: original stop() would exec into a dead container
        try:
            await original(self, delete)
        finally:
            del self.prepare_logs_for_host

    DockerEnvironment.stop = stop


DATA_CACHE = HERE.parent / "data_cache"  # filled by prefetch_data.py; path without spaces


def install_fast_setup():
    """Cut per-rollout sandbox setup (measured 46 s mean, 272 s max with 64 concurrent rollouts).

    1. Prebuilt image: every task's Dockerfile is just `FROM <base>` + `WORKDIR /workdir`, which the base image
       already has. Point the task at the base image, so Harbor skips the per-rollout `docker compose build`
       (and the matching `--rmi local` image removal at teardown).
    2. Data cache: bind-mount <data_cache>/<BUCKET_PREFIX> read-only at /home/user/input. The task's own
       healthcheck (pull_bucket.py) then sees the files and skips its HF-bucket download. A task whose prefix is
       not cached (no .complete marker) keeps the normal download.
    """
    from harbor.environments.docker.docker import DockerEnvironment

    original_init = DockerEnvironment.__init__
    original_mounts = DockerEnvironment._write_mounts_compose_file

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        dockerfile = Path(self.environment_dir) / "Dockerfile"
        if dockerfile.exists():
            lines = [l.strip() for l in dockerfile.read_text().splitlines() if l.strip()]
            if len(lines) == 2 and lines[0].startswith("FROM ") and lines[1] == "WORKDIR /workdir":
                base = lines[0].split()[1]
                if not self.task_env_config.docker_image:
                    self.task_env_config.docker_image = base
                    # Harbor snapshots the image into compose's variables during __init__.
                    self._env_vars.prebuilt_image_name = base
                self._dataagent_prebuilt = self.task_env_config.docker_image == base

    def _write_mounts_compose_file(self):
        prefix = ((self.task_env_config.env or {}).get("BUCKET_PREFIX") or "").rstrip("/")
        src = DATA_CACHE / prefix if prefix else None
        if src is not None and (src / ".complete").exists():
            if not any(m.get("target") == "/home/user/input" for m in self._mounts if isinstance(m, dict)):
                self._mounts.append({"type": "bind", "source": str(src), "target": "/home/user/input",
                                     "read_only": True})
        return original_mounts(self)

    DockerEnvironment.__init__ = __init__
    DockerEnvironment._write_mounts_compose_file = _write_mounts_compose_file


def install_fast_metadata():
    """Check this fixed Docker host once; chown all writable log mounts in one exec."""
    import shlex
    import subprocess
    from harbor.environments.docker.docker import DockerEnvironment

    daemon_os = DockerEnvironment._detect_daemon_os()
    if daemon_os is not None:
        DockerEnvironment._detect_daemon_os = classmethod(lambda cls: daemon_os)
    result = subprocess.run(
        DockerEnvironment._engine_cmd("info", "--format", "{{json .SecurityOptions}}"),
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode == 0:
        DockerEnvironment._rootless_docker = "rootless" in result.stdout

    original = DockerEnvironment.prepare_logs_for_host

    async def prepare_logs_for_host(self):
        if self._is_windows_container or not hasattr(os, "getuid"):
            return await original(self)
        targets = self._mount_targets(writable_only=True)
        if not targets:
            return
        uid, gid = (0, 0) if await self._is_rootless_docker() else (os.getuid(), os.getgid())
        try:
            await self.service_exec(
                f"chown -R {uid}:{gid} " + " ".join(shlex.quote(t) for t in targets), user="root"
            )
        except Exception as exc:
            self.logger.warning(f"Failed to chown logs directory: {exc}")

    DockerEnvironment.prepare_logs_for_host = prepare_logs_for_host


def install_direct_exec():
    """Resolve each main container once and execute with Docker's lightweight CLI.

    Keep Harbor's shell, user, workdir, env merging, streaming and timeout handling.
    Compose remains responsible for container resources, mounts and networking.
    """
    import asyncio
    from harbor.constants import MAIN_SERVICE_NAME
    from harbor.environments.docker.docker import DockerEnvironment

    original = DockerEnvironment._compose_exec

    async def _compose_exec(self, command, *, service, cwd, env, timeout_sec, user):
        if self._is_windows_container or service != MAIN_SERVICE_NAME:
            return await original(self, command, service=service, cwd=cwd, env=env,
                                  timeout_sec=timeout_sec, user=user)
        container = getattr(self, "_direct_exec_container", None)
        if container is None:
            container = await self._platform._resolve_service_container(service)
            self._direct_exec_container = container
        args = ["exec"]
        if cwd:
            args.extend(["-w", cwd])
        for key, value in (env or {}).items():
            args.extend(["-e", f"{key}={value}"])
        if user is not None:
            args.extend(["-u", str(user)])
        args.extend([container, *self._platform.exec_shell_args(command)])
        process = await asyncio.create_subprocess_exec(
            *type(self)._engine_cmd(*args), stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        callback = self._output_callback()
        if callback is not None:
            return await self._collect_streamed_output(
                process, timeout_sec=timeout_sec, stdin_data=None, on_output=callback
            )
        return await self._collect_buffered_output(process, timeout_sec=timeout_sec)

    DockerEnvironment._compose_exec = _compose_exec


def install_docker_limit(limit):
    """Bound Compose up/down/kill subprocesses while leaving agent execs concurrent."""
    import asyncio
    from harbor.environments.docker.docker import DockerEnvironment

    gate = asyncio.Semaphore(limit)
    original = DockerEnvironment._run_docker_compose_command

    async def run(self, command, *args, **kwargs):
        if command and command[0] in {"up", "down", "kill", "build"}:
            async with gate:
                return await original(self, command, *args, **kwargs)
        return await original(self, command, *args, **kwargs)

    DockerEnvironment._run_docker_compose_command = run


def install_shared_network(name):
    """Reuse one dedicated Docker bridge for these single-container public-network tasks."""
    from harbor.environments.docker.docker import DockerEnvironment
    import subprocess

    check = subprocess.run(["docker", "network", "inspect", name],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if check.returncode != 0:
        subprocess.run(["docker", "network", "create", name], check=True)

    original = DockerEnvironment._write_resources_compose_file

    def write(self):
        path = original(self)
        if getattr(self, "_dataagent_prebuilt", False) and not self._enable_egress_control:
            config = json.loads(path.read_text())
            config["networks"] = {"default": {"name": name, "external": True}}
            path.write_text(json.dumps(config, indent=2))
        return path

    DockerEnvironment._write_resources_compose_file = write


def install_reprobe_degraded():
    """Do not cache a capture probe that came back below `tokens`.

    The proxy probes each engine+model once and caches the level forever. On 2026-10-11 the probe of
    trl-policy-v55 (fired while vLLM was busy with the adapter swap) returned `logprobs`, so every session
    for v55 was rejected ("training sampling requires an endpoint with token capture") and all four trainer
    resumes, which reuse v55, failed identically. Retry the probe a few times, and leave a degraded
    result uncached so the next session probes again.
    """
    import logging
    import time
    from openenv.core.harness.capture.server import UpstreamPool

    original = UpstreamPool._resolve_once
    log = logging.getLogger(__name__)

    def _resolve_once(self, upstream):
        for attempt in range(3):
            client, level = original(self, upstream)
            if level == "tokens":
                return client, level
            with self._lock:
                self._by_engine.pop(upstream.cache_key, None)
            log.warning("probe of %s gave %s (attempt %d); not cached", upstream.model, level, attempt + 1)
            time.sleep(5)
        return client, level

    UpstreamPool._resolve_once = _resolve_once


CONTEXT_EVENTS = HERE / "runs" / "context_events.jsonl"


def install_context_fixes(tool_cap, reserve_floor=256):
    """Two context-budget fixes in the capture proxy's upstream call (2026-10-13; FINDINGS.md C1, C2).

    1. Tool-output cap. Pi only truncates bash/read output at 50 KB (~15k tokens), more than the whole prompt
       budget, so a single wide-CSV `head` ended the rollout. Every tool message longer than `tool_cap` chars is
       cut to its head and tail with an explicit "[... N characters omitted ...]" marker before the request is
       rendered. Deterministic, so each turn's prompt still extends the previous one (prefix chain stays intact),
       and capture records exactly the tokens the model saw.
    2. Output reservation. The harness asks for max_tokens=4096 every call, so vLLM rejected any prompt above
       16,384 - 4,096 = 12,288 tokens. On that 400 the request is retried once with max_tokens shrunk to the room
       actually left (if at least `reserve_floor`); only then does the normal context-budget stop apply.
    Counters go to runs/context_events.jsonl (one line per event) for the training monitor.
    """
    import re
    import time
    from openenv.core.harness.capture import upstream as up

    original = up.InferenceClient.completion
    head = int(tool_cap * 0.7)
    tail = tool_cap - head

    def log(event, **kw):
        try:
            with CONTEXT_EVENTS.open("a") as f:
                f.write(json.dumps({"t": time.time(), "event": event, **kw}) + "\n")
        except OSError:
            pass

    def cap_text(text):
        if not isinstance(text, str) or len(text) <= tool_cap:
            return text, 0
        cut = len(text) - head - tail
        return f"{text[:head]}\n[... {cut} characters omitted ...]\n{text[-tail:]}", cut

    def cap_messages(messages):
        out, cuts = [], []
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "tool":
                content = m.get("content")
                if isinstance(content, str):
                    new, cut = cap_text(content)
                elif isinstance(content, list):
                    new, cut = [], 0
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            t, c = cap_text(part.get("text"))
                            new.append({**part, "text": t})
                            cut += c
                        else:
                            new.append(part)
                else:
                    new, cut = content, 0
                if cut:
                    m = {**m, "content": new}
                    cuts.append(cut)
            out.append(m)
        return out, cuts

    async def completion(self, request):
        req = dict(request)
        if isinstance(req.get("messages"), list):
            req["messages"], cuts = cap_messages(req["messages"])
            newest = req["messages"][-1] if req["messages"] else None
            # count each capped output once: on the turn where it is the newest message
            if cuts and isinstance(newest, dict) and newest.get("role") == "tool" \
                    and "characters omitted ...]" in str(newest.get("content")):
                log("tool_output_capped", omitted=cuts[-1])
        # vLLM gives max_completion_tokens precedence over max_tokens, so every output-limit field present
        # has to shrink together (changing only max_tokens left the request at 4096).
        keys = [k for k in ("max_tokens", "max_completion_tokens", "max_output_tokens") if k in req] or ["max_tokens"]
        key = keys[0]
        tried, errors = [], []
        while True:
            try:
                result = await original(self, req)
                if tried:
                    log("output_reservation_shrunk", max_tokens=req[key], attempts=len(tried))
                return result
            except up.UpstreamHTTPError as exc:
                msg = str(exc)
                if exc.status_code != 400 or "maximum context length" not in msg.lower():
                    raise
                limit = re.search(r"maximum context length is (\d+)", msg)
                prompt = re.search(r"prompt contains (?:at least )?(\d+)", msg)
                current = min(int(req.get(k) or 4096) for k in keys)
                tried.append(current)
                # An exact prompt count gives the room directly. vLLM usually reports only a lower bound
                # ("at least N" = limit - requested + 1, no information), so then step down a fixed ladder.
                exact = limit and prompt and "at least" not in msg[prompt.start() - 10:prompt.start() + 30]
                ladder = [int(limit.group(1)) - int(prompt.group(1)) - 8] if exact else []
                ladder += [3072, 2048, 1536, 1024, 768, 512, reserve_floor]
                nxt = next((x for x in ladder if reserve_floor <= x < current), None)
                errors.append(msg[msg.find("However"):][:160])
                if nxt is None:
                    log("context_exhausted", tried=tried, errors=errors)
                    raise
                for k in keys:
                    req[k] = nxt

    up.InferenceClient.completion = completion


def install_merge_split_assistant():
    """Re-join an assistant reply that a Responses-dialect harness (Codex) sends back as two messages.

    The model emits one turn: text, then tool calls. Codex returns it as a message item plus function_call items,
    which the dialect transform turns into two consecutive assistant messages (text-only, then content=None with
    tool_calls). The chat template then inserts `<|im_end|><|im_start|>assistant` between them, so the next prompt no
    longer extends the generated tokens and every turn becomes a separate training root (measured: 6 roots for 6
    turns). Merging restores the exact generated rendering. Other harnesses never send this pattern.
    """
    from openenv.core.harness.capture import upstream as up

    original = up.InferenceClient.completion

    def merge(messages):
        out = []
        for m in messages:
            prev = out[-1] if out else None
            if (isinstance(m, dict) and isinstance(prev, dict) and m.get("role") == "assistant"
                    and prev.get("role") == "assistant" and m.get("tool_calls") and not prev.get("tool_calls")
                    and not m.get("content") and isinstance(prev.get("content"), str)):
                out[-1] = {**m, "content": prev["content"]}
                continue
            out.append(m)
        return out

    async def completion(self, request):
        if isinstance(request.get("messages"), list):
            request = {**request, "messages": merge(request["messages"])}
        return await original(self, request)

    up.InferenceClient.completion = completion


def install_pi_tools(tools):
    """Restrict pi's tools with its own `--tools` allowlist (pi 1.0: "Comma-separated allowlist of tool names"),
    e.g. HARBOR_PI_TOOLS=bash: no read/edit/write, so data is inspected and the answer written via bash/python3.
    pi builds its system prompt from the enabled tools, so the read-specific rules disappear with the tool."""
    from harbor.agents.installed.pi import Pi

    original = Pi.build_cli_flags

    def build_cli_flags(self):
        flags = original(self) or ""
        return f"{flags} --tools {tools}".strip()

    Pi.build_cli_flags = build_cli_flags


def main():
    os.environ.setdefault("GOMAXPROCS", "2")
    install_reprobe_degraded()
    cap = int(os.environ.get("HARBOR_TOOL_OUTPUT_CAP", "0"))
    if cap > 0:
        install_context_fixes(cap)
    if os.environ.get("HARBOR_MERGE_SPLIT_ASSISTANT", "1") == "1":
        install_merge_split_assistant()
    os.environ.setdefault("PIRL_AGENT_VERSIONS", json.dumps({"opencode": OPENCODE_VERSION}))
    if os.environ.get("HARBOR_PI_TOOLS"):
        install_pi_tools(os.environ["HARBOR_PI_TOOLS"])
    register_prime_agent()
    install_opencode_prebaked_skip()
    install_mini_swe_prebaked_skip()
    install_fast_teardown()
    install_fast_setup()
    if os.environ.get("HARBOR_FAST_METADATA", "1") == "1":
        install_fast_metadata()
    if os.environ.get("HARBOR_DIRECT_EXEC", "0") == "1":
        install_direct_exec()
    ops = int(os.environ.get("HARBOR_DOCKER_OPS", "0"))
    if ops > 0:
        install_docker_limit(ops)
    shared = os.environ.get("HARBOR_SHARED_NETWORK", "dataagent-p3-fast")
    if shared:
        install_shared_network(shared)
    from pirl import harbor_server

    harbor_server.main()


if __name__ == "__main__":
    main()
