"""The Elo leaderboard: its records, their validation, and the rating fit.

Everything lives in `leaderboard/` at the repository root and is reviewed like code:

* `networks.json` -- the weights: file (relative to the shared `data/` tree), sha256, where it is
  published on Hugging Face, its behaviour report, a description and any older names.
* `players.json` -- a network plus the way it is run (tools/lib/player_spec.py), keyed by the
  player's canonical id; a bot has no network.
* `epochs.json` -- one engine epoch per entry: the engine fingerprints whose games it accepts, the
  main players, and the anchor that fixes the scale.
* `matches/<epoch>.jsonl` -- one line per pairing played, append-only: both seats' wins, losses
  and draws, the first deal seed, the engine and the commit. `.gitattributes` merges these files
  by union, so two branches that both appended merge cleanly; a record's `id` is a hash of its
  content, so a line that arrives twice is caught here.

The fit is Bradley-Terry with a seat term: P(a beats b, a as US) = logistic((r_a - r_b + h) / s),
s = 400 / ln 10, a draw half a win. It runs in two stages so that the main players' ratings do not
move when anyone else plays:

1. the main players, on main-vs-main pairings only, with the anchor pinned and h estimated;
2. everyone else, on every pairing that involves a non-main player (non-main against non-main
   included), with the main players and h held at stage 1's values.

A view ("main only", "main + lineage X") is a filter over this one fit, never a refit, so a
number means the same in every view. A non-main player is rated only if pairings connect it to
the main players. Standard library only: CI fits the leaderboard without torch or the engine.
"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, TypedDict

from tools.lib.player_spec import (Inference, PlayerSpecError, kind_of, player_id, spec_from_dict)

ROOT = pathlib.Path(__file__).resolve().parents[2]
LEADERBOARD_DIR = ROOT / "leaderboard"
HF_REPO = "mihaild/deepstruggle"
GITHUB_REPO = "mihaild/DeepStruggle"
#: Elo per natural-log unit of odds.
ELO_SCALE = 400.0 / math.log(10.0)
#: A weak Gaussian prior (standard deviation, Elo) on every fitted value, so that a player who
#: has won or lost every game still gets a finite rating. Against 2,000 games it moves a rating by
#: well under 0.1 Elo.
PRIOR_SD = 1000.0
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class Network(TypedDict):
    file: str
    sha256: str
    description: str
    old_names: List[str]
    hf: Optional[str]
    report: Optional[str]


class Player(TypedDict):
    network: Optional[str]
    inference: Dict[str, object]
    description: str


class Epoch(TypedDict):
    description: str
    engines: List[str]
    anchor: str
    anchor_elo: float
    main: List[str]


class Seat(TypedDict):
    w: int
    l: int
    d: int


class Record(TypedDict):
    id: str
    a: str
    b: str
    a_as_us: Seat
    a_as_ussr: Seat
    base_seed: int
    engine: str
    commit: str
    date: str


class LeaderboardError(ValueError):
    pass


@dataclass
class Board:
    networks: Dict[str, Network]
    players: Dict[str, Player]
    epochs: Dict[str, Epoch]
    matches: Dict[str, List[Record]]
    root: pathlib.Path = LEADERBOARD_DIR


def lineage(network: Optional[str]) -> Optional[str]:
    """The run a network descends from: its name up to the first `@` (`E7-A8-R1-S44@6400M+...`
    is lineage `E7-A8-R1-S44`)."""
    return network.split("@", 1)[0] if network else None


def player_spec(board: Board, pid: str) -> Inference:
    return spec_from_dict(board.players[pid]["inference"])


def player_lineage(board: Board, pid: str) -> Optional[str]:
    return lineage(board.players[pid]["network"])


# ---- reading and writing ----------------------------------------------------------------------


def _read_json(path: pathlib.Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise LeaderboardError(f"missing {path}") from None
    except json.JSONDecodeError as e:
        raise LeaderboardError(f"{path}: not JSON ({e})") from None


def load(root: pathlib.Path = LEADERBOARD_DIR) -> Board:
    networks = _read_json(root / "networks.json")
    players = _read_json(root / "players.json")
    epochs = _read_json(root / "epochs.json")
    if not (isinstance(networks, dict) and isinstance(players, dict) and isinstance(epochs, dict)):
        raise LeaderboardError("networks.json, players.json and epochs.json must each hold an object")
    matches: Dict[str, List[Record]] = {}
    for name in epochs:
        path = root / "matches" / f"{name}.jsonl"
        rows: List[Record] = []
        if path.exists():
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        raise LeaderboardError(f"{path}:{n}: not JSON ({e})") from None
        matches[name] = rows
    return Board(networks, players, epochs, matches, root)  # type: ignore[arg-type]


def _dump(obj: object) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False) + "\n"


def save_registry(board: Board) -> None:
    """Write networks.json, players.json and epochs.json, keys sorted so diffs stay small."""
    (board.root / "networks.json").write_text(_dump(dict(sorted(board.networks.items()))), encoding="utf-8")
    (board.root / "players.json").write_text(_dump(dict(sorted(board.players.items()))), encoding="utf-8")
    (board.root / "epochs.json").write_text(_dump(board.epochs), encoding="utf-8")


def record_id(rec: Mapping[str, object]) -> str:
    """A hash of everything in the record but its id."""
    body = {k: v for k, v in rec.items() if k != "id"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]


def make_record(a: str, b: str, a_as_us: Seat, a_as_ussr: Seat, base_seed: int, engine: str,
                commit: str, date: str) -> Record:
    body = {"a": a, "b": b, "a_as_us": dict(a_as_us), "a_as_ussr": dict(a_as_ussr),
            "base_seed": int(base_seed), "engine": engine, "commit": commit, "date": date}
    return {"id": record_id(body), **body}  # type: ignore[return-value]


def append_record(board: Board, epoch: str, rec: Record) -> None:
    """Append one pairing to the epoch's match file (and to `board`), refusing a repeat."""
    key = _pair_key(rec)
    for old in board.matches.setdefault(epoch, []):
        if old["id"] == rec["id"] or _pair_key(old) == key:
            raise LeaderboardError(f"{rec['a']} vs {rec['b']} at base seed {rec['base_seed']} is "
                                   f"already recorded ({old['id']})")
    path = board.root / "matches" / f"{epoch}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    board.matches[epoch].append(rec)


def _pair_key(rec: Record) -> Tuple[str, str, int]:
    """A pairing at a base seed plays the same deals whichever side is `a`, so this is what must
    be unique: a second record of it would count the same games twice."""
    x, y = sorted((rec["a"], rec["b"]))
    return x, y, int(rec["base_seed"])


# ---- validation -------------------------------------------------------------------------------


def _seat_ok(s: object) -> bool:
    return (isinstance(s, dict) and set(s) == {"w", "l", "d"}
            and all(isinstance(s[k], int) and not isinstance(s[k], bool) and s[k] >= 0 for k in s))


def validate(board: Board, repo_root: pathlib.Path = ROOT) -> List[str]:
    """Every problem in the records, as messages; empty when they are consistent."""
    errs: List[str] = []
    for name, net in board.networks.items():
        where = f"networks.json[{name}]"
        if "~" in name or ":" in name or "@" not in name:
            errs.append(f"{where}: a network name has an '@' and no '~' or ':'")
        want = {"file", "sha256", "description", "old_names", "hf", "report"}
        if set(net) != want:
            errs.append(f"{where}: fields must be {sorted(want)}, got {sorted(net)}")
            continue
        if not isinstance(net["sha256"], str) or not _HEX64.match(net["sha256"]):
            errs.append(f"{where}: sha256 must be 64 lowercase hex digits")
        if not isinstance(net["old_names"], list) or not all(isinstance(x, str) for x in net["old_names"]):
            errs.append(f"{where}: old_names must be a list of strings")
        if net["report"] is not None and not (repo_root / net["report"]).is_file():
            errs.append(f"{where}: report {net['report']} does not exist")
        if net["hf"] is not None and (not isinstance(net["hf"], str) or net["hf"].startswith("/")):
            errs.append(f"{where}: hf must be a path inside {HF_REPO}, or null")
    for pid, pl in board.players.items():
        where = f"players.json[{pid}]"
        if set(pl) != {"network", "inference", "description"}:
            errs.append(f"{where}: fields must be ['description', 'inference', 'network'], got {sorted(pl)}")
            continue
        try:
            spec = spec_from_dict(pl["inference"])
            canon = player_id(pl["network"], spec)
        except PlayerSpecError as e:
            errs.append(f"{where}: {e}")
            continue
        if canon != pid:
            errs.append(f"{where}: the canonical id of this network and inference is {canon!r}")
        if pl["network"] is not None and pl["network"] not in board.networks:
            errs.append(f"{where}: network {pl['network']!r} is not in networks.json")
    for name, ep in board.epochs.items():
        where = f"epochs.json[{name}]"
        want = {"description", "engines", "anchor", "anchor_elo", "main"}
        if set(ep) != want:
            errs.append(f"{where}: fields must be {sorted(want)}, got {sorted(ep)}")
            continue
        if not ep["engines"] or not all(isinstance(e, str) and _HEX64.match(e) for e in ep["engines"]):
            errs.append(f"{where}: engines must be a non-empty list of engine fingerprints")
        for pid in ep["main"]:
            if pid not in board.players:
                errs.append(f"{where}: main player {pid!r} is not in players.json")
        if len(set(ep["main"])) != len(ep["main"]):
            errs.append(f"{where}: a main player is listed twice")
        if ep["anchor"] not in ep["main"]:
            errs.append(f"{where}: the anchor {ep['anchor']!r} must be a main player")
        seen_ids: Set[str] = set()
        seen_pairs: Dict[Tuple[str, str, int], str] = {}
        for n, rec in enumerate(board.matches.get(name, []), 1):
            at = f"matches/{name}.jsonl:{n}"
            want_r = {"id", "a", "b", "a_as_us", "a_as_ussr", "base_seed", "engine", "commit", "date"}
            if not isinstance(rec, dict) or set(rec) != want_r:
                errs.append(f"{at}: fields must be {sorted(want_r)}")
                continue
            if rec["a"] == rec["b"]:
                errs.append(f"{at}: a player against itself")
            for side in ("a", "b"):
                if rec[side] not in board.players:
                    errs.append(f"{at}: player {rec[side]!r} is not in players.json")
            if not (_seat_ok(rec["a_as_us"]) and _seat_ok(rec["a_as_ussr"])):
                errs.append(f"{at}: a_as_us / a_as_ussr must be {{w, l, d}} of non-negative integers")
            elif sum(int(s[k]) for s in (rec["a_as_us"], rec["a_as_ussr"]) for k in ("w", "l", "d")) == 0:
                errs.append(f"{at}: no games")
            if rec["engine"] not in ep["engines"]:
                errs.append(f"{at}: engine {rec['engine'][:12]}… is not one of {name}'s")
            if rec["id"] != record_id(rec):
                errs.append(f"{at}: id {rec['id']} does not match its content ({record_id(rec)}); "
                            f"records are never edited")
            if rec["id"] in seen_ids:
                errs.append(f"{at}: record {rec['id']} appears twice")
            seen_ids.add(rec["id"])
            key = _pair_key(rec)
            if key in seen_pairs and seen_pairs[key] != rec["id"]:
                errs.append(f"{at}: {key[0]} vs {key[1]} at base seed {key[2]} is already recorded "
                            f"({seen_pairs[key]}) -- the same deals, counted twice")
            seen_pairs.setdefault(key, rec["id"])
    for name in board.matches:
        if name not in board.epochs:
            errs.append(f"matches/{name}.jsonl: no epoch {name!r} in epochs.json")
    return errs


# ---- the fit ----------------------------------------------------------------------------------


@dataclass
class Rating:
    player: str
    elo: float
    se: float
    main: bool
    games: int
    score: float
    opponents: int
    main_opponents: int


@dataclass
class EpochFit:
    epoch: str
    anchor: str
    anchor_elo: float
    seat_us: float
    seat_us_se: float
    ratings: List[Rating]
    #: Players with games that no pairing connects to the main players.
    unrated: List[str] = field(default_factory=list)


# One binomial: (a, b, seat sign, score of a, games). seat +1 = a played the US.
_Obs = Tuple[str, str, int, float, int]


def _observations(records: Iterable[Record]) -> List[_Obs]:
    out: List[_Obs] = []
    for r in records:
        for sign, s in ((1, r["a_as_us"]), (-1, r["a_as_ussr"])):
            n = s["w"] + s["l"] + s["d"]
            if n:
                out.append((r["a"], r["b"], sign, s["w"] + 0.5 * s["d"], n))
    return out


def _cholesky_solve_and_inverse_diag(m: List[List[float]], g: List[float]) -> Tuple[List[float], List[float]]:
    """x with m x = g, and diag(m^-1), for a symmetric positive definite m."""
    n = len(m)
    low = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            acc = m[i][j] - sum(low[i][k] * low[j][k] for k in range(j))
            if i == j:
                if acc <= 0:
                    raise LeaderboardError("the fit's information matrix is not positive definite")
                low[i][i] = math.sqrt(acc)
            else:
                low[i][j] = acc / low[j][j]

    def solve(b: List[float]) -> List[float]:
        y = [0.0] * n
        for i in range(n):
            y[i] = (b[i] - sum(low[i][k] * y[k] for k in range(i))) / low[i][i]
        x = [0.0] * n
        for i in reversed(range(n)):
            x[i] = (y[i] - sum(low[k][i] * x[k] for k in range(i + 1, n))) / low[i][i]
        return x

    x = solve(g)
    diag = []
    for i in range(n):
        e = [0.0] * n
        e[i] = 1.0
        diag.append(solve(e)[i])
    return x, diag


def _newton(free: Sequence[str], fixed: Mapping[str, float], obs: Sequence[_Obs], fit_seat: bool,
            seat: float, prior_mean: float) -> Tuple[Dict[str, float], float, Dict[str, float], float]:
    """Maximise the binomial log-likelihood plus the weak prior over the `free` ratings (and the
    seat term when `fit_seat`), the rest held at `fixed` / `seat`. Returns the ratings, the seat
    term, and their standard errors, all in Elo."""
    idx = {p: i for i, p in enumerate(free)}
    n_var = len(free) + (1 if fit_seat else 0)
    h_i = len(free)
    v = [prior_mean] * len(free) + ([0.0] if fit_seat else [])
    mu = [prior_mean] * len(free) + ([0.0] if fit_seat else [])
    inv_var = 1.0 / PRIOR_SD ** 2

    def rating(p: str, vec: List[float]) -> float:
        return vec[idx[p]] if p in idx else fixed[p]

    def seat_of(vec: List[float]) -> float:
        return vec[h_i] if fit_seat else seat

    def loglik(vec: List[float]) -> float:
        ll = -0.5 * inv_var * sum((x - m) ** 2 for x, m in zip(vec, mu))
        h = seat_of(vec)
        for a, b, sign, s, n in obs:
            x = (rating(a, vec) - rating(b, vec) + sign * h) / ELO_SCALE
            # log p = -log(1 + e^-x), log(1 - p) = -log(1 + e^x), computed stably
            ll += -s * _softplus(-x) - (n - s) * _softplus(x)
        return ll

    if n_var == 0:
        return {}, seat, {}, 0.0
    ll = loglik(v)
    for _ in range(200):
        grad = [-(x - m) * inv_var for x, m in zip(v, mu)]
        hess = [[inv_var if i == j else 0.0 for j in range(n_var)] for i in range(n_var)]
        h = seat_of(v)
        for a, b, sign, s, n in obs:
            x = (rating(a, v) - rating(b, v) + sign * h) / ELO_SCALE
            p = 1.0 / (1.0 + math.exp(-x))
            d: List[Tuple[int, float]] = []
            if a in idx:
                d.append((idx[a], 1.0 / ELO_SCALE))
            if b in idx:
                d.append((idx[b], -1.0 / ELO_SCALE))
            if fit_seat:
                d.append((h_i, sign / ELO_SCALE))
            w = n * p * (1.0 - p)
            for i, di in d:
                grad[i] += (s - n * p) * di
                for j, dj in d:
                    hess[i][j] += w * di * dj
        step, _ = _cholesky_solve_and_inverse_diag(hess, grad)
        t = 1.0
        while True:
            cand = [x + t * dx for x, dx in zip(v, step)]
            cand_ll = loglik(cand)
            if cand_ll >= ll - 1e-12 or t < 1e-6:
                break
            t *= 0.5
        v, ll = cand, cand_ll
        if max(abs(t * dx) for dx in step) < 1e-7:
            break
    else:
        raise LeaderboardError("the rating fit did not converge in 200 Newton steps")
    # Standard errors from the curvature at the optimum.
    hess = [[inv_var if i == j else 0.0 for j in range(n_var)] for i in range(n_var)]
    h = seat_of(v)
    for a, b, sign, s, n in obs:
        x = (rating(a, v) - rating(b, v) + sign * h) / ELO_SCALE
        p = 1.0 / (1.0 + math.exp(-x))
        d = [(idx[a], 1.0 / ELO_SCALE)] if a in idx else []
        if b in idx:
            d.append((idx[b], -1.0 / ELO_SCALE))
        if fit_seat:
            d.append((h_i, sign / ELO_SCALE))
        for i, di in d:
            for j, dj in d:
                hess[i][j] += n * p * (1.0 - p) * di * dj
    _, var = _cholesky_solve_and_inverse_diag(hess, [0.0] * n_var)
    ratings = {p: v[i] for p, i in idx.items()}
    ses = {p: math.sqrt(var[i]) for p, i in idx.items()}
    return ratings, seat_of(v), ses, (math.sqrt(var[h_i]) if fit_seat else 0.0)


def _softplus(x: float) -> float:
    return x + math.log1p(math.exp(-x)) if x > 0 else math.log1p(math.exp(x))


def _components(nodes: Iterable[str], edges: Iterable[Tuple[str, str]]) -> Dict[str, str]:
    parent = {n: n for n in nodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return {n: find(n) for n in parent}


def fit_epoch(board: Board, epoch: str) -> EpochFit:
    ep = board.epochs[epoch]
    records = board.matches.get(epoch, [])
    mains = list(ep["main"])
    main_set = set(mains)
    anchor, anchor_elo = ep["anchor"], float(ep["anchor_elo"])

    # Stage 1: the main players among themselves.
    main_recs = [r for r in records if r["a"] in main_set and r["b"] in main_set]
    comp = _components(mains, ((r["a"], r["b"]) for r in main_recs))
    cut_off = sorted(p for p in mains if comp[p] != comp[anchor])
    if cut_off:
        raise LeaderboardError(f"{epoch}: no main-vs-main pairings connect these main players to the "
                               f"anchor {anchor}: {', '.join(cut_off)}")
    free1 = [p for p in mains if p != anchor]
    r1, seat, se1, seat_se = _newton(free1, {anchor: anchor_elo}, _observations(main_recs),
                                     fit_seat=bool(main_recs), seat=0.0, prior_mean=anchor_elo)
    elo: Dict[str, float] = {anchor: anchor_elo, **r1}
    se: Dict[str, float] = {anchor: 0.0, **se1}

    # Stage 2: everyone else, against the main players held fixed.
    other_recs = [r for r in records if not (r["a"] in main_set and r["b"] in main_set)]
    others = sorted({p for r in other_recs for p in (r["a"], r["b"])} - main_set)
    hub = "\0main"
    comp2 = _components([hub, *others],
                        ((hub if r["a"] in main_set else r["a"], hub if r["b"] in main_set else r["b"])
                         for r in other_recs))
    rated = [p for p in others if comp2[p] == comp2[hub]]
    unrated = [p for p in others if comp2[p] != comp2[hub]]
    rated_set = set(rated) | main_set
    obs2 = _observations(r for r in other_recs if r["a"] in rated_set and r["b"] in rated_set)
    r2, _, se2, _ = _newton(rated, {p: elo[p] for p in mains}, obs2, fit_seat=False, seat=seat,
                            prior_mean=anchor_elo)
    elo.update(r2)
    se.update(se2)

    # Per-player tallies over every record of the epoch.
    games: Dict[str, int] = {}
    score: Dict[str, float] = {}
    opps: Dict[str, Set[str]] = {}
    for r in records:
        for sign, s in ((1, r["a_as_us"]), (-1, r["a_as_ussr"])):
            n, pts = s["w"] + s["l"] + s["d"], s["w"] + 0.5 * s["d"]
            for p, q, got in ((r["a"], r["b"], pts), (r["b"], r["a"], n - pts)):
                games[p] = games.get(p, 0) + n
                score[p] = score.get(p, 0.0) + got
                opps.setdefault(p, set()).add(q)
    ratings = [Rating(player=p, elo=elo[p], se=se[p], main=p in main_set, games=games.get(p, 0),
                      score=score.get(p, 0.0), opponents=len(opps.get(p, ())),
                      main_opponents=len(opps.get(p, set()) & main_set))
               for p in elo]
    ratings.sort(key=lambda x: (-x.elo, x.player))
    return EpochFit(epoch, anchor, anchor_elo, seat, seat_se, ratings, unrated)


def view(board: Board, fit: EpochFit, *, main_only: bool = False,
         lineages: Sequence[str] = ()) -> List[Rating]:
    """The ratings a view shows: the main players, plus every player of the given lineages
    (all players when neither filter is given). A filter, never a refit."""
    if not main_only and not lineages:
        return list(fit.ratings)
    want = set(lineages)
    return [r for r in fit.ratings if r.main or player_lineage(board, r.player) in want]


# ---- the page's data --------------------------------------------------------------------------


def site_data(board: Board, commit: Optional[str] = None) -> Dict[str, object]:
    """Everything the leaderboard page shows, fitted: `leaderboard.json`, built by CI from the
    records rather than committed."""
    epochs: Dict[str, object] = {}
    for name, ep in board.epochs.items():
        fit = fit_epoch(board, name)
        pairs: Dict[Tuple[str, str], Dict[str, object]] = {}
        for r in board.matches.get(name, []):
            a, b = r["a"], r["b"]
            flip = a > b
            key = (b, a) if flip else (a, b)
            us, ussr = (r["a_as_ussr"], r["a_as_us"]) if flip else (r["a_as_us"], r["a_as_ussr"])
            # From the first player's side: their record as US and as USSR. Flipping a record swaps
            # who `a` is, so a's games as USSR become the new a's games as US, with w and l swapped.
            us_t = {"w": us["l"], "l": us["w"], "d": us["d"]} if flip else dict(us)
            ussr_t = {"w": ussr["l"], "l": ussr["w"], "d": ussr["d"]} if flip else dict(ussr)
            acc = pairs.setdefault(key, {"a": key[0], "b": key[1],
                                         "a_as_us": {"w": 0, "l": 0, "d": 0},
                                         "a_as_ussr": {"w": 0, "l": 0, "d": 0}, "records": []})
            for tgt, src in (("a_as_us", us_t), ("a_as_ussr", ussr_t)):
                for k in ("w", "l", "d"):
                    acc[tgt][k] += src[k]  # type: ignore[index]
            acc["records"].append(r["id"])  # type: ignore[union-attr]
        epochs[name] = {
            "description": ep["description"], "engines": ep["engines"], "anchor": ep["anchor"],
            "anchor_elo": ep["anchor_elo"], "main": ep["main"], "seat_us": round(fit.seat_us, 2),
            "seat_us_se": round(fit.seat_us_se, 2), "unrated": fit.unrated,
            "ratings": [{"player": r.player, "elo": round(r.elo, 1), "se": round(r.se, 1),
                         "main": r.main, "games": r.games, "score": r.score,
                         "opponents": r.opponents, "main_opponents": r.main_opponents}
                        for r in fit.ratings],
            "pairs": list(pairs.values()),
        }
    players = {}
    for pid, pl in board.players.items():
        spec = spec_from_dict(pl["inference"])
        players[pid] = {"network": pl["network"], "lineage": lineage(pl["network"]),
                        "kind": kind_of(spec), "inference": pl["inference"],
                        "description": pl["description"]}
    return {"schema": 1, "commit": commit, "hf_repo": HF_REPO, "github": GITHUB_REPO,
            "networks": board.networks, "players": players, "epochs": epochs}


def markdown_table(board: Board, ratings: Sequence[Rating]) -> str:
    rows = ["| # | player | Elo | ± | games | score | main opps | |", "|---:|:---|---:|---:|---:|---:|---:|:---|"]
    for k, r in enumerate(ratings, 1):
        pct = 100.0 * r.score / r.games if r.games else 0.0
        rows.append(f"| {k} | `{r.player}` | {r.elo:.0f} | {r.se:.0f} | {r.games:,} | {pct:.1f}% | "
                    f"{r.main_opponents} | {'main' if r.main else ''} |")
    return "\n".join(rows)
