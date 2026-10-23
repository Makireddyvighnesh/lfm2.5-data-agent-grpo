"""Harbor agent for Prime Agent (Prime Intellect's pi fork), wired to OpenEnv's capture proxy.

Harbor 0.23 has no Prime Agent adapter. This mirrors OpenEnv's `InterceptPi`: the seam hands over
`intercept_config` (proxy URL, session id as API key, model) and setup() writes a pi-style
models.json into ~/.prime/agent/. Sessions are written to /logs/agent/pi/sessions because Prime Agent
keeps pi's session JSONL format, so OpenEnv's pi-session trace reader (patched in
SmolDataEnvs/pirl/harbor_server.py) cross-checks these rollouts the same way it does pi's.

The sandbox image (sandbox_image/Dockerfile) has Prime Agent 0.10.0 preinstalled; install() only
falls back to the official installer when it is missing.
"""

import json
import shlex
from typing import Any

from harbor.agents.installed.base import BaseInstalledAgent, with_prompt_template
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

VERSION = "0.10.0"
PROVIDER = "intercept"  # same provider alias as OpenEnv's pi seam, so model_fmt can be shared
PATH = 'export PATH="$HOME/.local/bin:$PATH"; '
MAX_TOKENS = 4096  # per-call output cap, same as pirl.common.MAX_TOKENS
CONTEXT = 32768  # vLLM --max-model-len


class InterceptPrimeAgent(BaseInstalledAgent):
    _OUTPUT_FILENAME = "prime-agent.txt"

    def __init__(self, *args: Any, intercept_config: dict[str, str] | None = None, **kwargs: Any):
        self._intercept_config = intercept_config or {}
        super().__init__(*args, **kwargs)

    @staticmethod
    def name() -> str:
        return "prime-agent"

    def get_version_command(self) -> str | None:
        return PATH + "prime-agent --version"

    def parse_version(self, stdout: str) -> str:
        return stdout.strip().splitlines()[-1].strip()

    async def install(self, environment: BaseEnvironment) -> None:
        check = await environment.exec(command=PATH + "prime-agent --version")
        if check.return_code == 0 and (check.stdout or "").strip().endswith(VERSION):
            self._version = VERSION
            return
        await self.exec_as_agent(
            environment,
            command=f"set -euo pipefail; curl -fsSL https://app.primeintellect.ai/prime-agent/install.sh "
            f"| PRIME_AGENT_VERSION={VERSION} sh && {PATH}prime-agent --version",
        )

    async def setup(self, environment: BaseEnvironment) -> None:
        await super().setup(environment)
        base_url = self._intercept_config.get("base_url")
        api_key = self._intercept_config.get("api_key")
        model = self._intercept_config.get("model")
        if not (base_url and api_key and model):
            raise ValueError("InterceptPrimeAgent requires intercept_config with base_url, api_key, model")
        models_json = json.dumps({"providers": {PROVIDER: {
            "baseUrl": f"{base_url}/v1",
            "api": "openai-completions",  # chat completions: the proxy's best-trodden dialect
            "apiKey": api_key,
            "models": [{"id": model, "contextWindow": CONTEXT, "maxTokens": MAX_TOKENS}],
        }}})
        await self.exec_as_agent(
            environment,
            command=f"mkdir -p ~/.prime/agent && printf '%s' {shlex.quote(models_json)} > ~/.prime/agent/models.json",
        )

    @with_prompt_template
    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        if not self.model_name or "/" not in self.model_name:
            raise ValueError("Model name must be in the format provider/model_name")
        provider, model_id = self.model_name.split("/", 1)
        await self.exec_as_agent(
            environment,
            command=(
                f"{PATH}cd /workdir 2>/dev/null || true; "
                f"prime-agent --print --mode json --offline "
                f"--session-dir /logs/agent/pi/sessions "
                f"--provider {shlex.quote(provider)} --model {shlex.quote(model_id)} "
                f"{shlex.quote(instruction)} "
                f"2>&1 </dev/null | grep -v '\"type\":\"message_update\"' "
                f"| stdbuf -oL tee /logs/agent/{self._OUTPUT_FILENAME}"
            ),
            # No telemetry from evaluation sandboxes.
            env={"PRIME_AGENT_TELEMETRY": "0", "DO_NOT_TRACK": "1"},
        )

    def populate_context_post_run(self, context: AgentContext) -> None:
        output = self.logs_dir / self._OUTPUT_FILENAME
        if not output.exists():
            return
        tokens_in = tokens_out = 0
        for line in output.read_text().splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            message = event.get("message") or {}
            if event.get("type") == "message_end" and message.get("role") == "assistant":
                usage = message.get("usage") or {}
                tokens_in += int(usage.get("input") or 0) + int(usage.get("cacheRead") or 0)
                tokens_out += int(usage.get("output") or 0)
        context.n_input_tokens = tokens_in
        context.n_output_tokens = tokens_out
