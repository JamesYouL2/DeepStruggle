"""P33f: does the trunk carry a belief about the opponent's hidden hand?

At positions of a network's own self-play, every card in the hidden pool -- the opponent's unseen hand
and the draw deck, from the decider's side, which the observation shows alike as "deck or hidden" --
is labelled by whether it is in the opponent's hand. Four predictors of that label, each read on
held-out games:

* the prior: the opponent's unseen hand size over the pool size, the same for every pool card;
* a public probe on the observation's card block and globals: where every card visibly is, the
  hand and deck counts, the turn -- what can be known without the board or any history;
* a probe on the whole observation: everything the network is shown;
* a probe on the network's frozen trunk (`extract_features`): what the network has represented.

If the trunk beats the public probe, a belief is already represented and a belief input would add
little; if it does not, the gap is what a history or belief feature could lift. The representation
decision is the owner's (research/plans/P33_search_preserved_gains.md); this only measures.

    PYTHONPATH=.:build/release python -m ai.eval.belief_probe --checkpoint swa.pt [--games 1024]
"""

from __future__ import annotations

import argparse
import copy
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

import ts_engine as ts
from bindings.ts_env import model_obs_features
from tools.lib.player_agent import NeuralAgent

CARD_OFFSET = 84 * 26
CARD_WIDTH = 110 * 14
GLOBAL_OFFSET = CARD_OFFSET + CARD_WIDTH
GLOBAL_WIDTH = 100


@dataclass
class Sample:
    obs: np.ndarray          # the decider's observation (base layout)
    label: np.ndarray        # 110: the card is in the opponent's hand (unseen by the decider)
    pool: np.ndarray         # 110: the card is in the hidden pool (opponent's unseen hand or the deck)
    game: int


def collect(model: torch.nn.Module, device: torch.device, games: int, every: int, temperature: float,
            seed: int) -> List[Sample]:
    """Self-play positions of `model`, one decision in `every`, each with its pool and labels."""
    runner = ts.VectorizedBatchRunner(games, seed)
    feats = model_obs_features(model)
    if feats:
        runner.set_obs_features([feats] * games, [feats] * games)
    rng = np.random.default_rng(seed)
    done = np.zeros(games, dtype=bool)
    out: List[Sample] = []
    for step in range(6000):
        done |= np.asarray(runner.get_terminals(), dtype=bool)
        if done.all():
            break
        obs = np.asarray(runner.get_observations(), dtype=np.float32)
        masks = np.asarray(runner.get_action_masks())
        players = np.asarray(runner.get_decision_players())
        with torch.no_grad():
            lg = model(torch.from_numpy(obs).to(device), torch.from_numpy(masks).to(device))[0].float()
            legal = torch.from_numpy(masks).to(device) > 0
            legal[~legal.any(-1), 0] = True               # a finished game has no legal move; its pick is ignored
            lg = lg.masked_fill(~legal, float("-inf"))
            acts = (torch.multinomial(torch.softmax(lg / temperature, -1), 1).squeeze(-1) if temperature > 0
                    else lg.argmax(-1)).cpu().numpy()
        for i in np.flatnonzero(~done):
            if int(players[i]) == 0 or rng.random() >= 1.0 / every:
                continue
            st = runner.get_state(int(i))
            me = ts.Player.US if int(players[i]) == 1 else ts.Player.USSR
            opp_unknown = int(ts.CardLocation.HAND_USSR_UNKNOWN if me == ts.Player.US else ts.CardLocation.HAND_US_UNKNOWN)
            locs = np.asarray([int(st.get_card_location(c)) for c in range(1, 111)])
            label = (locs == opp_unknown).astype(np.float32)
            pool = ((locs == opp_unknown) | (locs == int(ts.CardLocation.DRAW_DECK))).astype(np.float32)
            if label.sum() == 0:
                continue
            out.append(Sample(obs[i, :GLOBAL_OFFSET + GLOBAL_WIDTH].copy(), label, pool, int(i)))
        acts[done] = 0
        runner.step_flat_all([int(a) for a in acts], True)
    return out


def _trunk(model: torch.nn.Module, obs: torch.Tensor) -> torch.Tensor:
    """The network's trunk vector for `obs` (`extract_features`, which can also return attention)."""
    h = model.extract_features(obs)  # type: ignore[operator]
    assert isinstance(h, torch.Tensor)
    return h.float()


class _Probe(nn.Module):
    def __init__(self, width: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, 110))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def prior_logit(y: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """logit(hand size / pool size) for every card: the offset every probe starts from, so that each
    has to learn only what goes beyond that arithmetic."""
    p = (y.sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp(min=1)).clamp(1e-4, 1 - 1e-4)
    return torch.log(p / (1 - p)).expand_as(y)


def _masked_loss(probe: _Probe, x: torch.Tensor, y: torch.Tensor, m: torch.Tensor) -> float:
    with torch.no_grad():
        logit = probe(x) + prior_logit(y, m)
        return float((nn.functional.binary_cross_entropy_with_logits(logit, y, reduction="none") * m).sum() / m.sum())


def fit_probe(x: torch.Tensor, y: torch.Tensor, m: torch.Tensor, epochs: int, seed: int,
              val: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> _Probe:
    """A one-hidden-layer probe of the correction to the prior's logit, cross-entropy over the pool
    cards only; the epoch with the lowest loss on `val` (games held out of training) is kept, the
    untrained probe -- a zero correction, the prior itself -- included."""
    torch.manual_seed(seed)
    probe = _Probe(x.shape[1]).to(x.device)
    last = probe.net[-1]
    assert isinstance(last, nn.Linear)
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-5)
    best = (_masked_loss(probe, *val), copy.deepcopy(probe.state_dict()))
    n = x.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n, device=x.device)
        for lo in range(0, n, 512):
            idx = perm[lo:lo + 512]
            logit = probe(x[idx]) + prior_logit(y[idx], m[idx])
            loss = (nn.functional.binary_cross_entropy_with_logits(logit, y[idx], reduction="none") * m[idx]).sum() / m[idx].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        v = _masked_loss(probe, *val)
        if v < best[0]:
            best = (v, copy.deepcopy(probe.state_dict()))
    probe.load_state_dict(best[1])
    return probe.eval()


def _auc(p: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    npos, nneg = float(y.sum()), float(len(y) - y.sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def scores(p: np.ndarray, y: np.ndarray, m: np.ndarray) -> Tuple[float, float]:
    """Log loss and AUC over the pool entries."""
    sel = m > 0.5
    q = np.clip(p[sel], 1e-6, 1 - 1e-6)
    yy = y[sel]
    ll = float(-(yy * np.log(q) + (1 - yy) * np.log(1 - q)).mean())
    return ll, _auc(p[sel], yy)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--games", type=int, default=1024)
    ap.add_argument("--every", type=int, default=3, help="sample one decision in this many")
    ap.add_argument("--temperature", type=float, default=1.0, help="self-play sampling temperature")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=9091)
    a = ap.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralAgent.from_checkpoint(a.checkpoint, device=str(dev)).model.eval()
    samples = collect(model, dev, a.games, a.every, a.temperature, a.seed)
    test_games = set(range(0, a.games, 5))                       # one game in five held out
    val_games = set(range(1, a.games, 10))                       # and one in ten for early stopping
    va = [s for s in samples if s.game in val_games]
    tr = [s for s in samples if s.game not in test_games and s.game not in val_games]
    te = [s for s in samples if s.game in test_games]

    def stack(ss: List[Sample]) -> Dict[str, torch.Tensor]:
        obs = torch.from_numpy(np.stack([s.obs for s in ss])).to(dev)
        with torch.no_grad():
            h = torch.cat([_trunk(model, obs[lo:lo + 4096]) for lo in range(0, obs.shape[0], 4096)])
        return {"obs": obs, "pub": torch.cat([obs[:, CARD_OFFSET:CARD_OFFSET + CARD_WIDTH],
                                              obs[:, GLOBAL_OFFSET:GLOBAL_OFFSET + GLOBAL_WIDTH]], dim=1),
                "trunk": h, "y": torch.from_numpy(np.stack([s.label for s in ss])).to(dev),
                "m": torch.from_numpy(np.stack([s.pool for s in ss])).to(dev)}
    dtr, dva, dte = stack(tr), stack(va), stack(te)
    y, m = dte["y"].cpu().numpy(), dte["m"].cpu().numpy()
    prior = (y.sum(1, keepdims=True) / np.maximum(m.sum(1, keepdims=True), 1)) * m
    rows: List[Tuple[str, np.ndarray]] = [("prior: hand size / pool size", prior)]   # each probe corrects it
    for key, title in (("pub", "public probe: card block + globals"), ("obs", "probe on the whole observation"),
                       ("trunk", "probe on the frozen trunk")):
        probe = fit_probe(dtr[key], dtr["y"], dtr["m"], a.epochs, a.seed, (dva[key], dva["y"], dva["m"]))
        with torch.no_grad():
            p = torch.sigmoid(probe(dte[key]) + prior_logit(dte["y"], dte["m"])).cpu().numpy()
        rows.append((title, p))
    print(f"checkpoint {a.checkpoint}: {len(tr):,} training and {len(te):,} held-out positions "
          f"({a.games} games at T={a.temperature:g}, one decision in {a.every}); pool entries "
          f"{int(m.sum()):,}, of which in the opponent's hand {100 * y[m > 0.5].mean():.1f}%")
    base_ll = scores(prior, y, m)[0]
    for title, p in rows:
        ll, auc = scores(p, y, m)
        print(f"  {title:38s} log loss {ll:.4f} ({100 * (base_ll - ll) / base_ll:+5.1f}% vs the prior)  AUC {auc:.3f}")
    # per card: the trunk probe's AUC against the public probe's, where the card is often in the pool
    p_pub, p_tr = rows[1][1], rows[3][1]
    per: List[Tuple[float, int, float, float]] = []
    for c in range(110):
        sel = m[:, c] > 0.5
        if sel.sum() < 200 or y[sel, c].sum() < 20:
            continue
        a_pub, a_tr = _auc(p_pub[sel, c], y[sel, c]), _auc(p_tr[sel, c], y[sel, c])
        per.append((a_tr - a_pub, c + 1, a_pub, a_tr))
    per.sort()
    names = {c: str(ts.CardData.get_card_info(c)["name"]) for _, c, _, _ in per}
    print("  cards where the trunk knows most beyond public information (AUC trunk vs public):")
    for d, c, ap_, at in per[-8:][::-1]:
        print(f"    {names[c]:32s} {at:.3f} vs {ap_:.3f} ({d:+.3f})")
    print(f"  mean per-card AUC: public {np.mean([x[2] for x in per]):.3f}, trunk {np.mean([x[3] for x in per]):.3f} "
          f"over {len(per)} cards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
