# Deploying the web workbench

The workbench runs **in the browser**: the engine is compiled to WebAssembly, models run as ONNX
in onnxruntime-web, and the game (stepping, dice, undo, the action log, links) is TypeScript in the
page. There are two ways to publish it.

## GitHub Pages -- no server at all

`.github/workflows/pages.yml` builds the engine (Emscripten 6.0.10, pinned) and the page on every
push to `main` and publishes `web/ui/dist`. Enable it once: *Settings → Pages → Source: GitHub
Actions*. The page is then at `https://<owner>.github.io/<repo>/`.

With no server, models come from a **Hugging Face** model repo (type `owner/repo`, press *List*)
or an `.onnx` **dropped on the page**; replays from *Load Replay* or a dropped `.tslog.json`. A
link that names no model loads the most recently uploaded `.onnx` in `mihaild/deepstruggle`
(`DEFAULT_HF_REPO` in `web/ui/src/analysis/model.ts`), so publishing a new default model is an
upload there. Make the `.onnx` files with

```bash
PYTHONPATH=.:build/release .venv/bin/python tools/export_onnx.py \
  --checkpoint data/checkpoints/<run>/snapshot_final.pt --out <run>.onnx
```

which verifies the export against torch and records what the page checks (action view, observation
width, engine fingerprint) inside the file.

## Docker from this checkout, with nothing from the repository run as root

`web/Dockerfile` builds the same image from the **local checkout** (the build context is the
repository root; `web/Dockerfile.dockerignore` keeps `data/`, build directories, virtualenvs and
`node_modules` out of it), and it is written so that no repository file is ever executed as root:
root only installs distro packages and creates a user, before any source is copied in; the
WebAssembly build, the page's `npm ci` and `npm run build`, the clang build of the native engine,
`pip`, every script under `tools/` and the server run as unprivileged users. A hostile commit gets
a build container's user account and the runtime's `app` user, never uid 0 in any stage.

```bash
docker build -f web/Dockerfile -t deepstruggle-web .
docker run -d --name deepstruggle -p 127.0.0.1:8000:8000 \
  --read-only --tmpfs /tmp --cap-drop ALL --security-opt no-new-privileges \
  -v "$PWD/data/checkpoints:/data/checkpoints:ro" \
  -v deepstruggle-replays:/data/replays -v deepstruggle-onnx:/data/onnx_cache \
  deepstruggle-web
```

The run line adds what the image cannot enforce for the host: the root filesystem read-only (the
server writes only under `/data`), no capabilities, no privilege escalation, and the port
published on the host's loopback only -- there is no authentication. Pass
`--build-arg UID=$(id -u) --build-arg GID=$(id -g)` so the exports and replays the server writes
under `/data` belong to you; `--build-arg ENGINE_MARCH=native` when the image runs where it was
built. The runtime settings below apply unchanged.

## Docker -- with this machine's checkpoints

`Dockerfile` builds the page, its WebAssembly engine and the local files server **from the GitHub
repository** (`https://github.com/mihaild/DeepStruggle`, branch `main` by default), not from the
local checkout. What it deploys is whatever that ref holds: publish first, then build.

```bash
docker build -t deepstruggle-web deploy/web                      # main
docker build -t deepstruggle-web --build-arg REF=<sha> deploy/web  # a specific commit or tag
docker run -d --name deepstruggle -p 8000:8000 \
  -v /path/to/checkpoints:/data/checkpoints:ro \
  -v deepstruggle-replays:/data/replays \
  deepstruggle-web
# or: CHECKPOINTS=/path/to/checkpoints docker compose -f deploy/web/compose.yaml up -d --build
```

Then open `http://<host>:8000/`. The analysis panel lists the mounted checkpoints; the server
exports one to ONNX the first time it is picked (a few seconds, verified against torch) and caches
it in `/data/onnx_cache`.

Needs BuildKit (the default builder since Docker 23) for the git source. Pass the directory as
the context, as above: the Dockerfile uses nothing from it, and building from the repository
root would send the whole `data/` tree to the daemon.

### What is in the image

| stage | does |
|:---|:---|
| `source` | `ADD <repo>#<ref>`: BuildKit resolves the ref to a commit every build, so a new push rebuilds and an unchanged branch hits the cache |
| `wasm` | `emscripten/emsdk:6.0.10` builds the page's engine (`ts_engine_wasm`) with the sources' fingerprint baked in |
| `ui` | `npm ci && npm run build` → `web/ui/dist`, engine included |
| `engine` | clang builds the native `ts_engine`, which the ONNX export uses |
| runtime | `python:3.14-slim`, CPU torch + onnx/onnxscript/onnxruntime, fastapi/uvicorn; runs `python -m web.server.main` as uid 1000 |

**`-march`.** The repository compiles the native engine with `-march=native`, which suits a
training box and not an image: built on one CPU, it can die with SIGILL on another. The engine
stage swaps it, in the image's copy only, for `x86-64-v2` (`armv8-a` on ARM). Pass
`--build-arg ENGINE_MARCH=native` when the image runs on the machine that builds it. (The page's
engine is WebAssembly and has no `-march`.)

### Runtime settings

| variable | default in the image | meaning |
|:---|:---|:---|
| `TS_CHECKPOINTS_DIR` | `/data/checkpoints` | checkpoints the analysis panel offers; mount yours here |
| `TS_ONNX_CACHE_DIR` | `/data/onnx_cache` | exported models, reused until the checkpoint or the engine changes |
| `TS_REPLAYS_DIR` | `/data/replays` | replays the page can list and watch |
| `TS_DATA_ROOT` | `/data` | the shared data tree, so nothing resolves it through git |

A checkpoint's action view (E4 or E4.1) is read from its run directory's `metadata.json`, so
mount run directories whole, not bare `.pt` files.

There is no authentication. The server only serves the page and lists and exports the mounted
files, but anyone who can reach the port can read them. Put it behind a reverse proxy with auth
before exposing it.
