# Data Agent sandbox: SmolDataEnvs' pi image + Prime Agent and OpenCode preinstalled, so no harness
# is installed per rollout. Retagged as savatar101/env-data-agent-train:base (the tasks' FROM).
# Rollback: docker tag pirl/sandbox:pi-1.0.0 savatar101/env-data-agent-train:base
FROM pirl/sandbox:pi-1.0.0
ENV PRIME_AGENT_VERSION=0.10.0
RUN curl -fsSL https://app.primeintellect.ai/prime-agent/install.sh | sh \
 && ~/.local/bin/prime-agent --version
RUN bash -c ". ~/.nvm/nvm.sh && nvm use 22 && npm i -g opencode-ai@1.18.31 && opencode --version"
