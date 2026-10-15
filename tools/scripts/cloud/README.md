# Running training on rented GPU hosts

Provider-agnostic: everything needs only an SSH target, so it works on Vast.ai, RunPod or any box
you can reach. It served the E3-era P11 programme ([plan index](../../../research/plans/README.md);
the plan itself is in git history).

**The measurements below are from the E3 era (2026-09, ~13k steps/s on an RTX 4090).** Training
throughput has since grown several-fold, so the absolute numbers are stale; the conclusions about
what to filter on and how hosts fail are not.

## What to rent

* **One run per GPU.** Two runs on one 4090 gave 7,231 + 5,912 = 13,143 steps/s against 13,169
  solo (time-slicing), and CUDA MPS gave 12,484 combined -- no speedup. The GPU was not saturated
  (8 of 24 GB, 325 of 450 W); the engine is a CPU-side simulator. Running two in parallel is fine
  for scheduling, just not faster.
* **CPU is a first-class filter.** A solo run used 2.7 cores sustained, and the same 3090 ran
  5,072 steps/s on a 13.7-core host against 7,336 on a 16+ core host (45%). On Vast the field is
  `cpu_cores_effective`.
* **VRAM is not binding** (8 GB per run). Choose on **$ per run** (price per hour / steps per
  hour), not $/hr or VRAM.
* **Compare arms on the same GPU model.** TF32 and kernel selection differ between cards, a
  directional confound. `remote_launch.sh` records the GPU in `hardware.json`.
* **Filter on the driver**: `cuda_max_good>=13.0` (the image ships torch 2.13.0+cu130). Without
  it a host with an old driver rents, pulls the image, builds, and only then fails at CUDA init.

```
vastai search offers 'gpu_name=RTX_4090 num_gpus=1 cpu_cores_effective>=8 cpu_ram>=32 \
    disk_space>=100 reliability>0.98 inet_down>=1000 cuda_max_good>=13.0' -o 'dph+'
```

Measured with `bench_gpu.sh` at `num_envs` 512 everywhere (batch size changes the estimator):

| GPU | steps/s (E3) |
|:---|---:|
| RTX 4090 | 12,200 (local reference) |
| RTX 4080 | 9,348 |
| RTX 3090 | 5,072 – 7,336 (same card, different hosts) |
| RTX A4000 | 5,546 (saturated at 512) |

Pricing traps: renting four hosts of one model means paying the *fourth*-cheapest price (an A4000
went from $0.71 to $8.12 per arm); `dph_total` is the bill, not the search price (storage is
extra). A managed tier (RunPod Secure 4090, ~$2.51/arm with an SLA) was about the price actually
paid on a marketplace 3090 that also ran 60% slower.

## The scripts

```bash
tools/scripts/cloud/provision.sh <offer-id>          # rent, prepare, verify, leave running (ADOPT=<id> to pick one up)
tools/scripts/cloud/remote_bootstrap.sh root@1.2.3.4 # on any SSH host: ship source, build, prove the engine imports
tools/scripts/cloud/remote_launch.sh root@1.2.3.4 <run-name> -- <tools/train.py flags>   # detached, hardware recorded
tools/scripts/cloud/remote_sync.sh root@1.2.3.4 <run-name>                              # bring it home, continuously
KEEP=<id,...> tools/scripts/cloud/leak_guard.sh [max-minutes]     # warns about instances still billing
tools/scripts/cloud/bench_gpu.sh <offer-id> <label> [max-minutes] # rent, measure throughput, destroy
```

Derive the launch flags from a healthy run (`tools/scripts/launch_flags.py`, CLAUDE.md
invariant 15), not from an example. Then watch the *synced* local directory as for a local run:
`tools/scripts/watch_run.py data/checkpoints/<run-dir> --target-steps <N>`.

**Teardown: trust `leak_guard.sh`, not the trap.** `bench_gpu.sh` destroys its instance on exit,
but not under `pkill -TERM` while blocked in `ssh`; the instance kept billing and the guard
caught it. Run the guard whenever an instance exists.

## What comes home

`remote_sync.sh` brings metrics, TensorBoard, `hardware.json`, `train.log`, every snapshot, the
newest `resume_state.pt`, and a step-tagged resume about every `RESUME_STRIDE` steps (80M by
default; `--all-resume` for all, `RESUME_STRIDE=0` for none). Resume files are what lets an
experiment branch from a partial result. Selection is by spacing, not modulo: real step counts
(5046272, 40042496, ...) are multiples of nothing.

## Interruption

Spot instances get reclaimed. Training writes a resume state beside every snapshot, so an
interruption costs at most one snapshot interval. Re-bootstrap a replacement, push back the newest
`resume_*.pt`, and relaunch with `--resume`; `--train-steps` is the **total** step count, not the
remaining increment.

## The image and marketplace reliability

`Dockerfile` (and `provision.sh`) use `pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime`, the same
torch as the local venv (its arch list includes sm_120, so RTX 5090s work), plus cmake and clang.
The source is not baked in: experiment changes need an rsync, not an image rebuild. Benchmark on
this image only; an earlier bench on a cu124 torch failed on every 5090 and measured a stack no
training host used.

Provisioning four hosts once took nine rentals. Failures seen: an image pull unfinished after 35
minutes, a host stuck in `loading`, sshd closing every connection, a disk fault unpacking the
image, and an old driver (the only one visible in the offer). A working host reaches ssh in about
7 minutes, so **fail fast** at ~15 and re-rent.

The one GPU-selection attempt spent $0.106 and got no number: both hosts died in setup. It was
dropped because the 4090 was already near the cheapest per step and `torch.compile` promised more
than any GPU swap.
