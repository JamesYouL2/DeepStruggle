# Deep Struggle

[![tests](https://github.com/mihaild/DeepStruggle/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/mihaild/DeepStruggle/actions/workflows/tests.yml?query=branch%3Amain)
[![typecheck](https://github.com/mihaild/DeepStruggle/actions/workflows/typecheck.yml/badge.svg?branch=main)](https://github.com/mihaild/DeepStruggle/actions/workflows/typecheck.yml?query=branch%3Amain)
[![web](https://github.com/mihaild/DeepStruggle/actions/workflows/web.yml/badge.svg?branch=main)](https://github.com/mihaild/DeepStruggle/actions/workflows/web.yml?query=branch%3Amain)
[![pages](https://github.com/mihaild/DeepStruggle/actions/workflows/pages.yml/badge.svg?branch=main)](https://github.com/mihaild/DeepStruggle/actions/workflows/pages.yml)

Welcome to Deep Struggle! It is a project dedicated to build an AI that plays the Deluxe Edition of **Twilight Struggle** (GMT Games, 2005).

Current status:

* engine in C++ that mostly reproduces rules - some bugs are probably still there, but supposedly nothing major
* interface to plumb different player implementations, record games between them, replay them in web viewer, and run tournamnets
* network architectures and an RL training stack (NashPG self-play against an opponent pool), plus search at play time; the best models are on the leaderboard below, and what each training choice was worth is written up in [`research/`](research/README.md)

## Play, and the leaderboard

* **[Play against the models](https://mihaild.github.io/DeepStruggle/)** in the browser (GitHub
  Pages; no install, no server): choose *Play as US* or *Play as USSR* and the model takes the other
  side, or use the developer view to see every model's move probabilities and evaluation.
* **[Elo leaderboard](https://mihaild.github.io/DeepStruggle/leaderboard.html)** of the published
  models, with each one's head-to-head results and a behaviour report.
* **Models** are on Hugging Face: [`mihaild/deepstruggle`](https://huggingface.co/mihaild/deepstruggle).

> **About the Elo numbers.** The leaderboard's Elo is computed only from games between these
> models, on a scale anchored at one of them (`E7-A4-R1-S44@4800M` = 1500). It does **not**
> correspond to the human Elo ratings of the Playdek Twilight Struggle app or any other human
> rating: a model rated 1600 here is not a 1600 player there, and no game against a human
> contributes to it.

## AI Slop warning

This repository is developed with heavy usage of AI. Most code is reviewed, but not all of it, neither documentation. Be even more skeptical code quality here than usually.

Known, reproducible, unfixed defects are tracked in [`BUGS.md`](BUGS.md).

---

## Highlights

* **C++20 Simulation Core (`engine/`)**: Zero-allocation, high-throughput simulation engine covering the Deluxe Edition—all 110 cards, persistent board effects, simultaneous headline resolution, space race tracks, and regional scoring. Every card has an event handler, but that is not the same as every rule being right: known rules defects are tracked in [`BUGS.md`](BUGS.md).
* **Neural Policies & Reinforcement Learning (`ai/`)**: `LadderNet` (`ai/models/ladder_net.py`, a configurable trunk with per-entity policy heads) trained by Nash Policy Gradient (NashPG) self-play, optional behavioural cloning from self-play or recorded human games, search at play time (`ai/search/`: a Gumbel root and a rollout root), and a suite of evaluation probes in `ai/eval/`.
* **Interactive Web Workbench (`web/`)**: A Vite + TypeScript SVG Deluxe Map that runs entirely in the browser -- the engine compiled to WebAssembly, models run as ONNX -- for watching replays (`.tslog.json`), testing the engine by playing it, and playing with a model that shows its move probabilities and critic on every position. Works from GitHub Pages with no server; a small local server adds this machine's checkpoints and replays.
* **Evaluation & Tournament Suite (`tools/`)**: High-throughput tournament runner with Bradley-Terry Maximum Likelihood Elo estimation, automated loss-cause diagnostics, and match playback.

---

## Architecture Overview

```
.
├── engine/         # Core C++20 zero-allocation simulation engine
├── bindings/       # Native Python nanobind module (ts_engine) & environment bridge
├── ai/             # Networks (LadderNet; ColdWarNet V1/V2 for old checkpoints), rewards, NashPG, search, probes
├── bot/            # Baseline bots (HeuristicBot, RandomBot, StrategicBot, NeuralBot)
├── web/            # Web Workbench, in the browser (WebAssembly engine + ONNX models) + a local files server
├── tools/          # Unified CLI suite for training, tournaments, matches, and replays
├── rules/          # Formal specifications (cards, map topology, effect flags)
├── research/       # The research record: arm registry, findings, logs, plans
├── leaderboard/    # Elo leaderboard records (networks, players, matches)
└── tests/          # Python test suite across bindings, engine rules, and training
```

For in-depth architectural and developer documentation, see [`AGENTS.md`](AGENTS.md) and [`tools/README.md`](tools/README.md).

---

## Quickstart

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/mihaild/DeepStruggle.git
cd DeepStruggle

# Set up Python virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For a CUDA build of PyTorch, install it from [pytorch.org](https://pytorch.org) first; the rest
of `requirements.txt` installs on top of whatever `torch` is already present.

### 2. Build C++ Engine

The engine is built with clang and link-time optimisation (`apt-get install clang llvm lld`);
CMake finds them and refuses other compilers.

```bash
cmake -B build/release -S . -DPython_EXECUTABLE=$(pwd)/.venv/bin/python3
cmake --build build/release -j
```

### 3. Build Web UI & Launch Workbench

```bash
# Build the engine as WebAssembly and the page (needs Emscripten: tools/scripts/install_emsdk.sh)
tools/scripts/build_web.sh

# Serve it, with this machine's checkpoints and replays
PYTHONPATH=.:build/release .venv/bin/python -m web.server.main --port 8000
```

Open `http://localhost:8000` in your browser to launch the Web Workbench.

---

## Running Tests

Build the engine first. `pytest` compares the `ts_engine` it actually imported against the
`engine/` and `bindings/` sources and aborts the run if it is stale, since a plausible answer
from yesterday's rules is worse than a failure. With no extension built at all it says nothing —
the tests that need one fail on their own import, with a clearer message.

```bash
# Engine rules, the nanobind surface, and the training stack. ~1 min with -n auto.
PYTHONPATH=.:build/release python -m pytest -n auto tests/bindings tests/engine_logic tests/training
```

Two suites need something extra and are worth knowing about before you run a bare `pytest`:

```bash
# tests/replayer converts a corpus of real human games: downloaded once per machine into the
# git-ignored data tree, ~5 MB. A missing corpus fails these tests rather than skipping them.
PYTHONPATH=. python tools/download_ts_replayer.py
PYTHONPATH=.:build/release python -m pytest -n auto tests/replayer

# tests/web needs the built page (with its WebAssembly engine) and a browser, neither of which is
# in the repository. Emscripten: tools/scripts/install_emsdk.sh
tools/scripts/build_web.sh
python -m playwright install chromium
PYTHONPATH=.:build/release python -m pytest tests/web
```

`tests/differential/` cross-checks against an external reference engine. It is work in progress,
is excluded from collection, and is not part of the check a change is expected to pass.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Disclaimer

*Twilight Struggle* is a registered trademark of GMT Games LLC and was designed by Ananda Gupta and Jason Matthews. This repository is an independent research implementation developed strictly for academic, educational, and game-theoretic study. It is not affiliated with, authorized by, or endorsed by GMT Games LLC.
