# syntax=docker/dockerfile:1
#
# The web workbench from THIS checkout -- the page with its WebAssembly engine, plus the local files
# server that lists and exports this machine's checkpoints -- built so that nothing from the
# repository ever runs as root. Root does only one thing in any stage: `apt-get install` of distro
# packages, before any repository file is copied in. Every compile, every `npm` and `pip` step,
# every script under tools/ and the server itself run as an unprivileged user, so a hostile commit
# gets a build container's user account and the runtime's, never uid 0 anywhere.
#
#     docker build -f web/Dockerfile -t deepstruggle-web .
#     docker run -d --name deepstruggle -p 127.0.0.1:8000:8000 \
#         --read-only --tmpfs /tmp --cap-drop ALL --security-opt no-new-privileges \
#         -v "$PWD/data/checkpoints:/data/checkpoints:ro" \
#         -v deepstruggle-replays:/data/replays -v deepstruggle-onnx:/data/onnx_cache \
#         deepstruggle-web
#
# The build context is the repository root; web/Dockerfile.dockerignore keeps data/, build
# directories, virtualenvs and node_modules out of it. Compare deploy/web/Dockerfile, which builds
# the published GitHub ref instead of the checkout.
#
#     --build-arg UID=<uid> --build-arg GID=<gid>   the runtime user; match the owner of the mounted
#                                                  checkpoints so ONNX exports and replays it writes
#                                                  under /data belong to you (default 1000:1000)
#     --build-arg ENGINE_MARCH=<clang -march>       see the engine stage (default: portable)
#
# Checkpoints are not in the image. Mount a checkpoints tree laid out like data/checkpoints
# (<run>/snapshot_*.pt, with the run's metadata.json) at /data/checkpoints, read-only; the server
# exports one to ONNX the first time the page asks for it and caches it in /data/onnx_cache.

ARG PYTHON_IMAGE=python:3.14-slim
ARG NODE_IMAGE=node:22-slim
# Pinned like tools/scripts/install_emsdk.sh: the browser engine is checked bit for bit against
# the native one with this compiler (tests/web/test_wasm_engine.py).
ARG EMSDK_IMAGE=emscripten/emsdk:6.0.10
ARG UID=1000
ARG GID=1000

# -- the engine for the page (WebAssembly) -------------------------------------------------------
FROM ${EMSDK_IMAGE} AS wasm
# Root here only makes the user and its source directory; the toolchain is the image's. The uid is
# the system's pick: it owns nothing outside this stage. (WORKDIR creates a directory as root even
# after USER, so the directory is made and handed over here.)
RUN useradd --create-home build && mkdir /src && chown build:build /src
USER build
# Emscripten's cache of system libraries lives in the SDK directory, which is root's: give the
# build user its own (the first build fills it, a minute or so).
ENV EM_CACHE=/home/build/em_cache
WORKDIR /src
# What the fingerprint hashes (engine/, bindings/, the root CMakeLists.txt) and what the build reads.
COPY --chown=build:build CMakeLists.txt ./
COPY --chown=build:build engine ./engine
COPY --chown=build:build bindings ./bindings
COPY --chown=build:build tools ./tools
RUN emcmake cmake -B build/wasm -S . -DCMAKE_BUILD_TYPE=Release \
        -DTS_ENGINE_FINGERPRINT="$(PYTHONPATH=/src python3 tools/lib/engine_fingerprint.py)" \
 && cmake --build build/wasm --target ts_engine_wasm -j"$(nproc)"

# -- web UI ---------------------------------------------------------------------------------------
FROM ${NODE_IMAGE} AS ui
# The node image ships an unprivileged `node` user; npm and the page's build run as it. Root only
# makes the source directory over (WORKDIR would create it root-owned even after USER).
RUN mkdir -p /src/web/ui && chown -R node:node /src
USER node
WORKDIR /src/web/ui
COPY --chown=node:node web/ui/package.json web/ui/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY --chown=node:node web/ui/ ./
# The page bundles rules/map.json and rules/cards.json from outside web/ui.
COPY --chown=node:node rules /src/rules
COPY --chown=node:node --from=wasm /src/web/ui/public/engine ./public/engine
RUN npm run build

# -- ts_engine (native, for the ONNX export) -----------------------------------------------------
FROM ${PYTHON_IMAGE} AS engine
# The repository's CMake files compile with -march=native, right for a training box and wrong for
# an image: built on one CPU, it can die with SIGILL on another. The engine stage swaps it, in the
# image's copy only, for a portable baseline (x86-64-v2: any x86 server since ~2009; armv8-a on
# ARM). Pass ENGINE_MARCH=native when the image is built on the machine it runs on.
ARG ENGINE_MARCH=
# Root: distro packages only. clang, because the engine is built with clang and CMake refuses any
# other compiler; build-essential for make, libstdc++ and libgomp, which the batch runner links.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential clang cmake \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home build && mkdir /src && chown build:build /src
USER build
ENV PATH=/home/build/venv/bin:$PATH
# nanobind is a build-time dependency only; CMake finds it through the interpreter.
RUN python -m venv /home/build/venv && pip install --no-cache-dir "nanobind>=2.15.0"
WORKDIR /src
COPY --chown=build:build CMakeLists.txt ./
COPY --chown=build:build engine ./engine
COPY --chown=build:build bindings ./bindings
RUN march="${ENGINE_MARCH}"; \
    if [ -z "$march" ]; then \
      case "$(uname -m)" in \
        x86_64) march=x86-64-v2 ;; \
        aarch64|arm64) march=armv8-a ;; \
        *) march=native ;; \
      esac; \
    fi; \
    echo "ts_engine: -march=$march"; \
    sed -i "s/-march=native/-march=$march/" CMakeLists.txt engine/CMakeLists.txt \
 && cmake -B build/release -S . -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=/home/build/venv/bin/python \
 && cmake --build build/release --target ts_engine -j"$(nproc)" \
 && mkdir /home/build/out && cp build/release/ts_engine*.so /home/build/out/

# -- runtime --------------------------------------------------------------------------------------
FROM ${PYTHON_IMAGE}
ARG UID
ARG GID
# Root: distro packages and the user. libgomp, because the batch runner calls GCC's OpenMP runtime
# directly, under any compiler.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --gid ${GID} app && useradd --create-home --uid ${UID} --gid ${GID} app \
 && mkdir -p /data/checkpoints /data/replays /data/onnx_cache /app \
 && chown -R app:app /data /app
USER app
ENV PATH=/home/app/venv/bin:$PATH
# Only what the server imports: fastapi/uvicorn, and for exporting a checkpoint to ONNX torch (the
# CPU wheel: an export needs no GPU, and the CUDA wheels would add gigabytes), onnx/onnxscript, and
# onnxruntime, which verifies every export against torch before it is served. The versions are
# requirements.txt's.
RUN python -m venv /home/app/venv \
 && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.13.0" \
 && pip install --no-cache-dir "numpy>=2.5.0" "fastapi>=0.141.0" "uvicorn[standard]>=0.52.0" \
        "pydantic>=2.13.0" "onnx>=1.23.0" "onnxscript>=0.7.2" "onnxruntime>=1.30.0"
WORKDIR /app
COPY --chown=app:app . ./
COPY --chown=app:app --from=engine /home/build/out/ ./build/release/
COPY --chown=app:app --from=ui /src/web/ui/dist ./web/ui/dist

ENV PYTHONPATH=/app:/app/build/release \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TS_DATA_ROOT=/data \
    TS_CHECKPOINTS_DIR=/data/checkpoints \
    TS_REPLAYS_DIR=/data/replays \
    TS_ONNX_CACHE_DIR=/data/onnx_cache

# The server writes only under /data (exports, replays); the image itself can be mounted
# read-only, as the run line above does.
VOLUME ["/data/replays", "/data/onnx_cache"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/local/info', timeout=4)"]
# 0.0.0.0 inside the container; publish it to the host's loopback (-p 127.0.0.1:8000:8000) unless
# the page is meant to be reachable from elsewhere -- there is no authentication.
CMD ["python", "-m", "web.server.main", "--host", "0.0.0.0", "--port", "8000", "--log-level", "INFO"]
