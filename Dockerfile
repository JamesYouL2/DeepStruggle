FROM nvidia/cuda:13.3.1-cudnn-runtime-ubuntu26.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv curl git \
        build-essential clang libclang-rt-dev cmake nodejs npm zsh xz-utils libnspr4 libnss3 libatk1.0-0t64 libatk-bridge2.0-0t64 libatspi2.0-0t64 \
        libxdamage1 libasound2t64 libcups2t64

# Install Claude Code CLI
RUN npm install -g @anthropic-ai/claude-code

# Create non-root user (optional but recommended)
RUN groupadd -g 1001 claude
RUN useradd -m -s /bin/zsh -u 1001 -g 1001 claude
ENV NPM_CONFIG_PREFIX=/home/claude/.npm-global
ENV PATH="/home/claude/.npm-global/bin:${PATH}"
RUN mkdir -p /home/claude/.npm-global && chown -R claude:claude /home/claude/.npm-global

USER claude

ENV EMSDK_DIR=/home/claude/opt/emsdk
RUN git clone --depth 1 https://github.com/emscripten-core/emsdk.git $EMSDK_DIR \
&& $EMSDK_DIR/emsdk install 6.0.10 && $EMSDK_DIR/emsdk activate 6.0.10

WORKDIR /workspace

# Set up Python venv
RUN python3 -m venv /workspace/.venv
ENV PATH="/workspace/.venv/bin:$PATH"

CMD ["/bin/zsh"]
