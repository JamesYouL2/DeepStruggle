#!/usr/bin/env bash
# Start one training run on a prepared host, detached, and record what hardware ran it.
#
#   remote_launch.sh <ssh-target> <run-name> -- <tools/train.py args...>
#
# Two things this does beyond `ssh nohup`:
#
# * **Records the GPU model and core count into the run directory.** Arms being compared must run
#   on the same GPU model -- TF32 and kernel selection differ -- and without this recorded at
#   launch there is no way to check afterwards whether a surprising result was a hardware
#   confound. `hardware.json` is small and travels home with the metrics.
# * **Uses setsid**, so the run survives the SSH session closing. A previous sweep was lost
#   because `nohup ... &` died with its parent.
#
# --train-steps is CUMULATIVE. A resumed run inherits its predecessor's step count, so budgeting
# the increment makes the loop exit before its first iteration while still writing a final
# checkpoint that looks complete.
#
# The run name and the output directory are set here: passing --run-name or --output-dir among
# the train.py arguments is refused rather than appended twice.
set -euo pipefail

TARGET="${1:?usage: remote_launch.sh <ssh-target> <run-name> -- <train.py args...>}"
NAME="${2:?missing run name}"
shift 2
[ "${1:-}" = "--" ] && shift
[ $# -gt 0 ] || { echo "no train.py arguments given" >&2; exit 2; }

# A run name is the grammar of research/method/run_nomenclature.md (ai/training/run_name.py):
# letters, digits and `-`, `.` (E4.1), `@` and `+` (branch points) -- nothing else, and never `..`.
# It becomes a remote directory, so a `/`, a quote or a space must never reach the remote shell.
if ! [[ "$NAME" =~ ^[A-Za-z0-9][A-Za-z0-9@+.-]*$ ]] || [[ "$NAME" == *..* ]]; then
    echo "remote_launch.sh: '$NAME' is not a run name (letters, digits, - . @ + only; see" \
         "research/method/run_nomenclature.md)" >&2
    exit 2
fi
for arg in "$@"; do
    case "$arg" in
        --run-name|--run-name=*|--output-dir|--output-dir=*)
            echo "remote_launch.sh: do not pass $arg -- the run name is the second argument and the" \
                 "output directory is derived from it" >&2
            exit 2 ;;
    esac
done

REMOTE_DIR="${REMOTE_DIR:-/workspace/ts}"

# `user@host:port` -> ssh -p port user@host. Every Vast host is reached on a high port through
# the proxy, and a bare `ssh "$TARGET"` silently tries 22. A bare user@host or an ssh-config
# alias still works unchanged.
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)
case "$TARGET" in
    *:*) SSH_OPTS+=(-p "${TARGET##*:}"); TARGET="${TARGET%:*}" ;;
esac

# The remote script is a quoted heredoc, so nothing in it is expanded locally; every value
# reaches it as a positional argument, each quoted with %q because ssh joins its arguments into
# one string that the remote shell parses again.
REMOTE_ARGS=()
for v in "$REMOTE_DIR" "$NAME" "$@"; do
    printf -v q '%q' "$v"
    REMOTE_ARGS+=("$q")
done

ssh "${SSH_OPTS[@]}" "$TARGET" bash -s -- "${REMOTE_ARGS[@]}" <<'REMOTE'
set -euo pipefail
REMOTE_DIR="$1"
NAME="$2"
shift 2
OUT="${REMOTE_DIR}/runs/${NAME}"
cd "$REMOTE_DIR"
mkdir -p "$OUT"

PY=/opt/venv/bin/python
[ -x "$PY" ] || PY=$(command -v python3.14 || command -v python3)

"$PY" - <<'PYEOF' > "$OUT/hardware.json"
import json, os, subprocess
def q(field):
    try:
        return subprocess.run(["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip().splitlines()[0]
    except Exception:
        return "unknown"
print(json.dumps({
    "gpu_name": q("name"),
    "gpu_memory_total": q("memory.total"),
    "driver_version": q("driver_version"),
    "cpu_count": os.cpu_count(),
    "host": os.uname().nodename,
}, indent=2))
PYEOF
echo "--- hardware ---"; cat "$OUT/hardware.json"

cd "$REMOTE_DIR"
setsid nohup env TRITON_CACHE_DIR=.triton_cache PYTHONPATH=.:build/release \
    "$PY" tools/train.py "$@" --run-name "$NAME" --output-dir "$OUT" \
    > "$OUT/train.log" 2>&1 < /dev/null &
sleep 5
echo "==> launched $NAME on $(hostname); log at $OUT/train.log"
REMOTE
