# SmolDataEnvs sandbox with Node 22 + pi preinstalled, so Harbor's per-rollout pi install is skipped.
FROM docker.io/savatar101/env-data-agent-train:base
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | env -u NODE_VERSION bash \
 && export NVM_DIR="$HOME/.nvm" && . "$NVM_DIR/nvm.sh" \
 && nvm install 22 && nvm alias default 22 \
 && npm install -g --ignore-scripts @earendil-works/pi-coding-agent@1.0.0 && pi --version
