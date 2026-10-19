"""P36c: fit a student's policy to the rollout root's improved policy -- signal or capacity?

`tools/distill_targets.py` writes the targets: on the network's own entry decisions, the keyed root's
improved policy (R32's form) at several world counts, held out by game, the held-out rows searched twice.
This trains a student on one world count's targets by cross-entropy, on the policy heads only (the value
heads frozen), **to convergence**: the stop is the training fit no longer rising, not the held-out loss --
the question is whether the student *can* fit the targets, which an early-stopped fit never answers (the
earlier offline rounds' two epochs at 1e-4 underfit by design). Students, after the plan: the current
network fine-tuned (`init`'s weights); the same shape from scratch (`reinit`); a larger shape (`reinit`
with `arch`).

Three numbers per evaluation, the plan's cell readings:

1. **training fit** -- on a fixed sample of training positions where the teacher departs (its target's top
   move is not the network's argmax), the share where the student's argmax is the teacher's top move;
2. **held-out agreement** -- the same on held-out games' departures (and on all held-out positions);
3. **the teacher's self-agreement** on those held-out positions -- its second seed's top move against its
   first, on the same departures: the ceiling (2) can reach.

The z2 picks are read alike (`*_pick`): where the root's own decision departs from the network's move.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import ts_engine as ts
from ai.models.ladder_net import create_ladder_net, ladder_config_from_state_dict
from ai.training.generic_trainer import _resolve_run_dir
from ai.training.value_fit import Codes, _git

ACTIONS = 220


@dataclass
class PolicyFitConfig:
    dataset: str                    # a tools/distill_targets.py .npz
    worlds: int                     # which world count's targets
    init: str                       # an RL checkpoint: its weights (fine-tune) or its shape (reinit)
    reinit: bool = False
    #: Changes to the init network's ladder config (LadderNet keyword -> value), with reinit only.
    arch: Optional[Dict[str, Any]] = None
    batch: int = 4096
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200
    max_epochs: float = 60.0
    eval_every: int = 200
    patience: int = 10              # evaluations without the training fit rising by min_gain
    min_gain: float = 0.002
    fit_sample: int = 20000         # training positions the fit is read on
    seed: int = 44
    device: str = "cuda"
    bf16: bool = True


def dense_targets(acts: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """(n, 16) sparse improved policies -> (n, 220) dense rows."""
    out = np.zeros((acts.shape[0], ACTIONS), dtype=np.float32)
    rows = np.repeat(np.arange(acts.shape[0], dtype=np.int64), acts.shape[1])
    a = acts.reshape(-1).astype(np.int64)
    p = probs.reshape(-1)
    ok = a >= 0
    out[rows[ok], a[ok]] = p[ok]            # a row's moves are distinct
    return out


def masks_of(states: np.ndarray) -> np.ndarray:
    out = np.empty((states.shape[0], ACTIONS), dtype=np.uint8)
    for i in range(states.shape[0]):
        out[i] = np.asarray(ts.ActionMask.generate_flat_mask(ts.state_from_raw_bytes(states[i].tobytes())))
    return out


def run_policy_fit(cfg: PolicyFitConfig, run_name: Optional[str], description: Optional[str],
                   output_dir: Optional[str] = None, cli: Optional[Dict[str, Any]] = None) -> str:
    """Fit to convergence, read the three numbers, keep the last and the best held-out; returns the run dir."""
    dev = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    sd = torch.load(cfg.init, map_location="cpu", weights_only=True)
    sd.pop("keep_idx", None)
    lcfg = ladder_config_from_state_dict(sd)
    if lcfg is None:
        raise SystemExit(f"{cfg.init}: policy fit takes a LadderNet checkpoint")
    if cfg.arch:
        if not cfg.reinit:
            raise SystemExit("a changed architecture (--policy-arch) trains from fresh weights: add --policy-reinit")
        unknown = sorted(set(cfg.arch) - set(lcfg))
        if unknown:
            raise SystemExit(f"--policy-arch: not ladder settings: {unknown}")
        lcfg = {**lcfg, **cfg.arch}
    model = create_ladder_net(dev, **lcfg)
    if not cfg.reinit:
        model.load_state_dict(sd, strict=True)
    # The value path frozen: the student is read and trained on its policy only.
    for name, p in model.named_parameters():
        if name.startswith(("val_win_head", "val_vp_head", "val_logits_head")):
            p.requires_grad_(False)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in model.parameters())

    t0 = time.time()
    with np.load(cfg.dataset) as z:
        d = {k: z[k] for k in z.files}
    w = cfg.worlds
    if f"acts_{w}" not in d:
        raise SystemExit(f"{cfg.dataset} has no targets at {w} worlds")
    heldout = d["heldout"].astype(bool)
    codes = Codes(int(getattr(model, "obs_feature_bits", 0)), dev)
    x_all = codes.encode(d["states"])
    m_all = masks_of(d["states"])
    y_all = dense_targets(d[f"acts_{w}"], d[f"probs_{w}"])
    top = d[f"acts_{w}"][:, 0].astype(np.int64)                 # the target's top move (sorted by probability)
    top2 = d[f"acts2_{w}"][:, 0].astype(np.int64)
    net = d["net_argmax"].astype(np.int64)
    pick, pick2 = d[f"pick_{w}"].astype(np.int64), d[f"pick2_{w}"].astype(np.int64)
    tr, ho = np.flatnonzero(~heldout), np.flatnonzero(heldout)
    dep_tr = tr[top[tr] != net[tr]]
    fit_rows = np.sort(rng.choice(dep_tr, size=min(cfg.fit_sample, dep_tr.size), replace=False)) if dep_tr.size else dep_tr
    dep_ho = ho[top[ho] != net[ho]]
    pdep_tr = tr[pick[tr] != net[tr]]
    pdep_ho = ho[pick[ho] != net[ho]]
    ceiling = float((top2[dep_ho] == top[dep_ho]).mean()) if dep_ho.size else float("nan")
    ceiling_all = float((top2[ho] == top[ho]).mean()) if ho.size else float("nan")
    ceiling_pick = float((pick2[pdep_ho] == pick[pdep_ho]).mean()) if pdep_ho.size else float("nan")
    print(f"policy fit at {w} worlds: {tr.size:,} train / {ho.size:,} held-out positions; teacher departures "
          f"{100 * dep_tr.size / max(1, tr.size):.1f}% / {100 * dep_ho.size / max(1, ho.size):.1f}%; the teacher's own "
          f"held-out agreement {100 * ceiling:.1f}% on departures, {100 * ceiling_all:.1f}% overall, z2 picks "
          f"{100 * ceiling_pick:.1f}%; {n_params / 1e6:.2f}M parameters; loaded in {time.time() - t0:.0f}s", flush=True)

    x_dev = torch.from_numpy(x_all).to(dev)
    m_dev = torch.from_numpy(m_all).to(dev)
    y_dev = torch.from_numpy(y_all).to(dev)
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, tr.size // cfg.batch)
    max_steps = int(cfg.max_epochs * steps_per_epoch)

    def lr_at(step: int) -> float:
        if step < cfg.warmup_steps:
            return cfg.lr * (step + 1) / cfg.warmup_steps
        frac = (step - cfg.warmup_steps) / max(1, max_steps - cfg.warmup_steps)
        return cfg.lr * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, frac))))

    def logits(rows: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.bf16 and dev.type == "cuda"):
            lg = model(codes.decode(x_dev[rows]), m_dev[rows])[0]
        return lg.float().masked_fill(m_dev[rows] == 0, float("-inf"))

    @torch.no_grad()
    def argmax_of(rows: np.ndarray) -> np.ndarray:
        model.eval()
        out = [logits(torch.from_numpy(rows[lo:lo + 8192]).to(dev)).argmax(-1).cpu().numpy()
               for lo in range(0, rows.size, 8192)]
        model.train()
        return np.concatenate(out) if out else np.zeros(0, np.int64)

    @torch.no_grad()
    def held_ce() -> float:
        model.eval()
        tot, n = 0.0, 0
        for lo in range(0, ho.size, 8192):
            r = torch.from_numpy(ho[lo:lo + 8192]).to(dev)
            lp = torch.log_softmax(logits(r), -1).nan_to_num(neginf=0.0)
            tot += float(-(y_dev[r] * lp).sum().item())
            n += int(r.numel())
        model.train()
        return tot / max(1, n)

    def evaluate() -> Dict[str, float]:
        out = {"held_ce": held_ce()}
        if fit_rows.size:
            out["train_fit"] = float((argmax_of(fit_rows) == top[fit_rows]).mean())
        if dep_ho.size:
            out["held_agree_dep"] = float((argmax_of(dep_ho) == top[dep_ho]).mean())
        out["held_agree_all"] = float((argmax_of(ho) == top[ho]).mean())
        if pdep_ho.size:
            out["held_agree_pick"] = float((argmax_of(pdep_ho) == pick[pdep_ho]).mean())
        if pdep_tr.size:
            sample = pdep_tr[:cfg.fit_sample]
            out["train_fit_pick"] = float((argmax_of(sample) == pick[sample]).mean())
        out.update(ceiling_dep=ceiling, ceiling_all=ceiling_all, ceiling_pick=ceiling_pick)
        return out

    run_dir = _resolve_run_dir(output_dir, run_name, "policyfit", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    meta: Dict[str, Any] = dict(cli or {})
    if not cfg.reinit:
        meta["resumed_from"] = cfg.init
    meta |= {"run_name": run_name, "mode": "policy-fit", "description": description, "seed": cfg.seed,
             "git_commit": _git("rev-parse", "HEAD"),
             "git_dirty": bool(_git("status", "--porcelain", "--", "ai", "tools", "bindings", "engine")),
             "dataset": cfg.dataset, "dataset_meta": json.loads(pathlib.Path(cfg.dataset + ".meta.json").read_text())
             if pathlib.Path(cfg.dataset + ".meta.json").exists() else None,
             "init": cfg.init, "config": asdict(cfg), "ladder_config": lcfg, "parameters": n_params,
             "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (pathlib.Path(run_dir) / "metadata.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    log = open(os.path.join(run_dir, "policy_fit_metrics.jsonl"), "a", encoding="utf-8")
    first = evaluate()
    first.update(step=0, epoch=0.0)
    print("  step 0:", {k: round(v, 4) for k, v in first.items()}, flush=True)
    log.write(json.dumps(first) + "\n")
    log.flush()
    best_fit, best_ce, bad, step, cursor = -1.0, math.inf, 0, 0, 0
    perm = rng.permutation(tr)
    t_fit = time.time()
    last: Dict[str, float] = first
    while step < max_steps:
        if cursor + cfg.batch > perm.size:
            perm, cursor = rng.permutation(tr), 0
        idx = torch.from_numpy(np.sort(perm[cursor:cursor + cfg.batch])).to(dev)
        cursor += cfg.batch
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        lp = torch.log_softmax(logits(idx), -1).nan_to_num(neginf=0.0)
        loss = -(y_dev[idx] * lp).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        step += 1
        if step % cfg.eval_every == 0 or step == max_steps:
            ev = evaluate()
            ev.update(step=step, epoch=round(step / steps_per_epoch, 2), lr=lr_at(step), train_ce=float(loss.item()),
                      seconds=round(time.time() - t_fit, 1))
            log.write(json.dumps(ev) + "\n")
            log.flush()
            last = ev
            print(f"  step {step} epoch {ev['epoch']}: train fit {100 * ev.get('train_fit', float('nan')):.1f}%  held-out "
                  f"{100 * ev.get('held_agree_dep', float('nan')):.1f}% on departures (teacher {100 * ceiling:.1f}%), "
                  f"{100 * ev['held_agree_all']:.1f}% overall (teacher {100 * ceiling_all:.1f}%)  held-out CE "
                  f"{ev['held_ce']:.4f}", flush=True)
            if ev["held_ce"] < best_ce:
                best_ce = ev["held_ce"]
                torch.save(model.state_dict(), os.path.join(run_dir, "policy_fit_best_heldout.pt"))
            if ev.get("train_fit", 0.0) > best_fit + cfg.min_gain:
                best_fit, bad = ev.get("train_fit", 0.0), 0
            else:
                bad += 1
                if bad >= cfg.patience:
                    print(f"  converged: the training fit has not risen by {cfg.min_gain} in {bad} evaluations", flush=True)
                    break
    torch.save(model.state_dict(), os.path.join(run_dir, "policy_fit_last.pt"))
    meta.update(finished=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), steps=step, initial=first, final=last)
    (pathlib.Path(run_dir) / "metadata.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    log.close()
    print(f"policy fit done at step {step}: train fit {100 * last.get('train_fit', float('nan')):.1f}%, held-out "
          f"{100 * last.get('held_agree_dep', float('nan')):.1f}% (teacher {100 * ceiling:.1f}%) -> {run_dir}")
    return run_dir


def counts(model_cfg: Dict[str, Any]) -> Tuple[int, int]:
    """Parameters of a ladder shape and of the policy path alone (for sizing students)."""
    m = create_ladder_net("cpu", **model_cfg)
    total = sum(p.numel() for p in m.parameters())
    value = sum(p.numel() for n, p in m.named_parameters() if n.startswith(("val_win_head", "val_vp_head", "val_logits_head")))
    return total, total - value
