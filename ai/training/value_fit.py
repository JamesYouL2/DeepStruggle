"""P34's supervised value fit: an evaluator trained on playout-labelled positions.

The dataset is tools/value_labels.py's (train.npz, holdout.npz, siblings.npz, manifest.json). The
network is a LadderNet, either fine-tuned from an RL checkpoint's weights or the same shape freshly
initialised; what is trained is its value path -- `v_win` toward the label's mean score, `v_vp`
toward the mean final VP (/20) -- plus a small auxiliary head on the same trunk features (VP and
DEFCON at the turn's end, each region's swing at its next scoring), kept beside the checkpoint and
not part of it. The sibling slice adds a paired loss: in each world the candidates' leaves were
played on the same deal and dice, so the difference of their results is a low-noise target for the
difference of their values (P33's value probe: the paired term is what lifted sibling correlation).

The output is an ordinary checkpoint -- the network's full state dict -- so the rollout root reads it
as a leaf network unchanged. Its policy heads are untouched by the fit and not meant to be played.

Observations are stored as exact one-byte codes (ts.ObsCodebook, one table per run) and decoded on
the GPU; the sibling leaves are encoded per batch from their raw states.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import ts_engine as ts
from ai.models.ladder_net import LadderNet, create_ladder_net, ladder_config_from_state_dict
from ai.training.generic_trainer import _resolve_run_dir

#: Region order of the dataset's swing columns (tools/value_labels.REGIONS).
N_REGIONS = 7


@dataclass
class ValueFitConfig:
    dataset: str
    init: str                       # an RL checkpoint: its weights (fine-tune) or its shape (--value-reinit)
    reinit: bool = False
    #: Changes to the init network's ladder config (LadderNet keyword -> value), with reinit only:
    #: the deep shape is {"hidden_dim": 768, "num_res_blocks": 8} (A3's), the token shape
    #: {"token_layers": 2, "token_dim": 128} (A6's).
    arch: Optional[Dict[str, Any]] = None
    batch: int = 4096
    sibling_groups: int = 256       # (position, world) groups of k leaves per step
    sibling_coef: float = 10.0
    vp_coef: float = 1.0
    aux_coef: float = 0.1
    lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 300
    max_epochs: float = 20.0
    eval_every: int = 250
    patience: int = 6               # evaluations without a better held-out loss
    seed: int = 44
    device: str = "cuda"
    bf16: bool = True


class AuxHead(nn.Module):
    """Turn-end VP (/20), turn-end DEFCON (/5) and the regions' swings (/10) from trunk features."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 128), nn.LayerNorm(128), nn.GELU(), nn.Linear(128, 2 + N_REGIONS))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


def _win_logit(model: LadderNet, h: torch.Tensor) -> torch.Tensor:
    """v_win's pre-tanh value z: v_win = tanh(z) and P(win) = sigmoid(2z), so the cross-entropy is
    taken on 2z without clamping a probability."""
    head = model.val_win_head
    if not isinstance(head, nn.Sequential) or not isinstance(head[-1], nn.Tanh):
        raise SystemExit("value fit needs a scalar v_win head ending in tanh (not a categorical model)")
    return head[:-1](h).squeeze(-1)


def _vp(model: LadderNet, h: torch.Tensor) -> torch.Tensor:
    head = model.val_vp_head
    if head is None:
        raise SystemExit("value fit needs a scalar v_vp head")
    return head(h).squeeze(-1)


def _load_split(path: pathlib.Path) -> Dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


class Codes:
    """The run's one codebook, its values on the device, and encoding of raw states."""

    def __init__(self, features: int, device: torch.device) -> None:
        self.book = ts.ObsCodebook()
        self.features = features
        self.device = device
        self.values = torch.zeros((256, 1), dtype=torch.float32, device=device)
        self._n = 0

    def encode(self, states: np.ndarray) -> np.ndarray:
        out = []
        for lo in range(0, states.shape[0], 65536):
            out.append(self.book.encode_raw(np.ascontiguousarray(states[lo:lo + 65536]), self.features))
        self._sync()
        return np.concatenate(out) if out else np.zeros((0, ts.OBS_SIZE), np.uint8)

    def _sync(self) -> None:
        if self.book.size() != self._n:
            v = self.book.values()
            self.values[:v.size, 0] = torch.from_numpy(v).to(self.device)
            self._n = int(v.size)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return F.embedding(codes.int(), self.values).squeeze(-1)


def _targets(d: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    swing = np.nan_to_num(d["swing"], nan=0.0) / 10.0
    return {
        "score": d["score"].astype(np.float32),
        "vp": d["final_vp"].astype(np.float32),
        "aux": np.concatenate([d["turn_vp"][:, None], (d["turn_defcon"] / 5.0)[:, None], swing], axis=1).astype(np.float32),
        "aux_w": np.concatenate([np.ones((d["score"].size, 2), np.float32), d["swing_frac"].astype(np.float32)], axis=1),
    }


def _sibling_groups(s: Dict[str, np.ndarray], holdout: bool) -> Tuple[np.ndarray, np.ndarray]:
    """(groups x k row indices, groups x k scores) for the (position, world) groups whose every
    candidate kept a leaf, in the requested split."""
    meta = s["meta"]
    keep = (s["holdout"] == holdout) & (meta[:, 2] == 0)
    key = s["id"] * 4096 + meta[:, 1]
    order = np.lexsort((meta[:, 0], key))
    order = order[keep[order]]
    k = int(meta[:, 0].max()) + 1 if meta.size else 0
    rows: List[np.ndarray] = []
    if k == 0:
        return np.zeros((0, 0), np.int64), np.zeros((0, 0), np.float32)
    keys = key[order]
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    ends = np.r_[starts[1:], keys.size]
    for a, b in zip(starts, ends):
        g = order[a:b]
        if g.size == k and (meta[g, 3] == 1).all():
            rows.append(g)
    idx = np.stack(rows) if rows else np.zeros((0, k), np.int64)
    return idx, s["cols"][idx, 0].astype(np.float32) if idx.size else np.zeros((0, k), np.float32)


def parse_arch(items: Optional[List[str]]) -> Dict[str, Any]:
    """`key=value` ladder settings: integers, floats, true/false, else strings."""
    out: Dict[str, Any] = {}
    for item in items or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise SystemExit(f"--value-arch takes key=value, got {item!r}")
        value: Any = raw
        if raw in ("true", "false"):
            value = raw == "true"
        else:
            for conv in (int, float):
                try:
                    value = conv(raw)
                    break
                except ValueError:
                    pass
        out[key] = value
    return out


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_value_fit(cfg: ValueFitConfig, run_name: Optional[str], description: Optional[str],
                  output_dir: Optional[str] = None, cli: Optional[Dict[str, Any]] = None) -> str:
    """Fit, evaluate on held-out games, keep the best; returns the run directory. `cli` is the
    tools/train.py settings, recorded top-level in metadata.json under their own names so
    launch_flags.py and run_codes.py read a value fit like any other run."""
    dev = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    ds = pathlib.Path(cfg.dataset)
    sd = torch.load(cfg.init, map_location="cpu", weights_only=True)
    sd.pop("keep_idx", None)
    lcfg = ladder_config_from_state_dict(sd)
    if lcfg is None:
        raise SystemExit(f"{cfg.init}: value fit takes a LadderNet checkpoint")
    if cfg.arch:
        if not cfg.reinit:
            raise SystemExit("a changed architecture (--value-arch) trains from fresh weights: add --value-reinit")
        unknown = sorted(set(cfg.arch) - set(lcfg))
        if unknown:
            raise SystemExit(f"--value-arch: not ladder settings: {unknown}")
        lcfg = {**lcfg, **cfg.arch}
    model = create_ladder_net(dev, **lcfg)
    if not cfg.reinit:
        model.load_state_dict(sd, strict=True)
    model.train()
    manifest = json.loads((ds / "manifest.json").read_text())
    engine = manifest["engine"]
    features = int(getattr(model, "obs_feature_bits", 0))
    width = model.TOTAL_OBS_SIZE

    t0 = time.time()
    codes = Codes(features, dev)
    train = _load_split(ds / "train.npz")
    hold = _load_split(ds / "holdout.npz")
    x_train = codes.encode(train.pop("state"))
    x_hold = codes.encode(hold.pop("state"))
    y_train, y_hold = _targets(train), _targets(hold)
    sib_path = ds / "siblings.npz"
    sib = _load_split(sib_path) if sib_path.exists() else None
    if sib is not None:
        g_train, gy_train = _sibling_groups(sib, holdout=False)
        g_hold, gy_hold = _sibling_groups(sib, holdout=True)
    else:
        g_train = g_hold = np.zeros((0, 0), np.int64)
        gy_train = gy_hold = np.zeros((0, 0), np.float32)
    print(f"value fit: {x_train.shape[0]:,} train / {x_hold.shape[0]:,} held-out positions, "
          f"{g_train.shape[0]:,} / {g_hold.shape[0]:,} sibling groups, codebook {codes.book.size()}, "
          f"loaded in {time.time() - t0:.0f}s", flush=True)

    aux = AuxHead(model.val_win_head[0].in_features).to(dev)   # type: ignore[index]
    params = list(model.parameters()) + list(aux.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, x_train.shape[0] // cfg.batch)
    max_steps = int(cfg.max_epochs * steps_per_epoch)

    def lr_at(step: int) -> float:
        if step < cfg.warmup_steps:
            return cfg.lr * (step + 1) / cfg.warmup_steps
        frac = (step - cfg.warmup_steps) / max(1, max_steps - cfg.warmup_steps)
        return cfg.lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, frac))))

    y_dev = {k: torch.from_numpy(v).to(dev) for k, v in y_train.items()}

    def heads(obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.bf16 and dev.type == "cuda"):
            h = model._encode(obs)[0]
            z = _win_logit(model, h)
            vp = _vp(model, h)
            a = aux(h)
        return z.float(), vp.float(), a.float()

    def leaf_obs(rows: np.ndarray) -> torch.Tensor:
        assert sib is not None
        c = codes.book.encode_raw(np.ascontiguousarray(sib["leaf"][rows.reshape(-1)]), features)
        codes._sync()
        return codes.decode(torch.from_numpy(c).to(dev, non_blocking=True))

    def paired(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(2.0 * z)
        return (((p - p.mean(1, keepdim=True)) - (y - y.mean(1, keepdim=True))) ** 2).mean()

    @torch.no_grad()
    def evaluate() -> Dict[str, float]:
        model.eval()
        aux.eval()
        zs, vps = [], []
        for lo in range(0, x_hold.shape[0], 8192):
            z, vp, _ = heads(codes.decode(torch.from_numpy(x_hold[lo:lo + 8192]).to(dev)))
            zs.append(z)
            vps.append(vp)
        z = torch.cat(zs)
        vp = torch.cat(vps)
        y = torch.from_numpy(y_hold["score"]).to(dev)
        yv = torch.from_numpy(y_hold["vp"]).to(dev)
        bce = F.binary_cross_entropy_with_logits(2.0 * z, y).item()
        # The floor: the labels' own entropy -- a perfect fit of the 32-world means reaches it.
        yc = y.clamp(1e-6, 1 - 1e-6)
        floor = (-(yc * yc.log() + (1 - yc) * (1 - yc).log())).mean().item()
        out = {"hold_bce": bce, "hold_bce_floor": floor, "hold_vp_mse": F.mse_loss(vp, yv).item(),
               "hold_score_mse": F.mse_loss(torch.sigmoid(2.0 * z), y).item(),
               "hold_score_var": float(y.var().item())}
        if g_hold.shape[0]:
            n = min(g_hold.shape[0], 4000)
            zz = []
            for lo in range(0, n, 512):
                zg, _, _ = heads(leaf_obs(g_hold[lo:min(n, lo + 512)]))
                zz.append(zg.view(-1, g_hold.shape[1]))
            zg = torch.cat(zz)
            yg = torch.from_numpy(gy_hold[:n]).to(dev)
            out["hold_sibling_paired_mse"] = paired(zg, yg).item()
            dz = zg[:, :, None] - zg[:, None, :]
            dy = yg[:, :, None] - yg[:, None, :]
            decided = dy != 0
            out["hold_sibling_pair_acc"] = ((dz.sign() == dy.sign()) & decided).sum().item() / max(1, decided.sum().item())
        model.train()
        aux.train()
        return out

    run_dir = _resolve_run_dir(output_dir, run_name, "valuefit", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    meta: Dict[str, Any] = dict(cli or {})
    if not cfg.reinit:
        meta["resumed_from"] = cfg.init      # the lineage: a fine-tune branches from its RL network
    meta |= {"run_name": run_name, "mode": "value-fit", "description": description, "seed": cfg.seed,
            "git_commit": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain", "--", "ai", "tools", "bindings", "engine")),
            "engine": engine, "dataset": manifest["dataset"], "dataset_dir": str(ds),
            "dataset_files": manifest["files"], "init": cfg.init, "config": asdict(cfg),
            "ladder_config": lcfg, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (pathlib.Path(run_dir) / "metadata.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    log = open(os.path.join(run_dir, "value_fit_metrics.jsonl"), "a", encoding="utf-8")

    first = evaluate()
    first.update(step=0, epoch=0.0)
    print("  step 0 (the initial network):", {k: round(v, 4) for k, v in first.items()}, flush=True)
    log.write(json.dumps(first) + "\n")
    log.flush()
    best = math.inf
    best_step = 0
    bad = 0
    step = 0
    perm = rng.permutation(x_train.shape[0])
    cursor = 0
    t_fit = time.time()
    while step < max_steps:
        if cursor + cfg.batch > perm.size:
            perm = rng.permutation(x_train.shape[0])
            cursor = 0
        idx = np.sort(perm[cursor:cursor + cfg.batch])
        cursor += cfg.batch
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        obs = codes.decode(torch.from_numpy(x_train[idx]).to(dev, non_blocking=True))
        ti = torch.from_numpy(idx).to(dev)
        z, vp, a = heads(obs)
        loss_win = F.binary_cross_entropy_with_logits(2.0 * z, y_dev["score"][ti])
        loss_vp = F.mse_loss(vp, y_dev["vp"][ti])
        w = y_dev["aux_w"][ti]
        loss_aux = (w * (a - y_dev["aux"][ti]) ** 2).sum() / w.sum().clamp(min=1.0)
        loss = loss_win + cfg.vp_coef * loss_vp + cfg.aux_coef * loss_aux
        loss_sib = torch.zeros((), device=dev)
        if g_train.shape[0] and cfg.sibling_coef > 0:
            gi = rng.integers(0, g_train.shape[0], size=cfg.sibling_groups)
            zg, _, _ = heads(leaf_obs(g_train[gi]))
            loss_sib = paired(zg.view(len(gi), -1), torch.from_numpy(gy_train[gi]).to(dev))
            loss = loss + cfg.sibling_coef * loss_sib
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        step += 1
        if step % cfg.eval_every == 0 or step == max_steps:
            ev = evaluate()
            score = ev["hold_bce"] + cfg.vp_coef * ev["hold_vp_mse"]
            ev.update(step=step, epoch=round(step / steps_per_epoch, 2), lr=lr_at(step),
                      train_win=loss_win.item(), train_vp=loss_vp.item(), train_aux=loss_aux.item(),
                      train_sib=loss_sib.item(), seconds=round(time.time() - t_fit, 1))
            log.write(json.dumps(ev) + "\n")
            log.flush()
            better = score < best - 1e-5
            print(f"  step {step} epoch {ev['epoch']}: held-out bce {ev['hold_bce']:.4f} (floor {ev['hold_bce_floor']:.4f}) "
                  f"vp {ev['hold_vp_mse']:.4f} pair-acc {ev.get('hold_sibling_pair_acc', float('nan')):.3f}"
                  f"{'  *' if better else ''}", flush=True)
            if better:
                best, best_step, bad = score, step, 0
                torch.save(model.state_dict(), os.path.join(run_dir, "value_fit_best.pt"))
                torch.save(aux.state_dict(), os.path.join(run_dir, "value_aux_best.pt"))
            else:
                bad += 1
                if bad >= cfg.patience:
                    print(f"  early stop: no better held-out loss in {bad} evaluations", flush=True)
                    break
    meta.update(finished=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), best_step=best_step,
                steps=step, best_holdout=best, initial=first)
    (pathlib.Path(run_dir) / "metadata.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    log.close()
    print(f"value fit done: best held-out at step {best_step} -> {run_dir}/value_fit_best.pt")
    return run_dir
