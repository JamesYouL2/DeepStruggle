"""What a player is: a network plus the way it is run.

The same weights make many players -- greedy, sampled at a temperature, under PUCT search, under a
Gumbel root at some budget -- and each is a different strength. The agent strings `load_agent`
takes (`temp:0.5:<ckpt>`, `gumbel:<ckpt>:256:8`, `search:<ckpt>:128:determinize:all::cpp:0`)
say this positionally, with the defaults in code, so the same string can come to mean a different
player when a default moves. Here each way of running a network is a typed spec with named fields,
and a player's id spells it out:

    E7-A4-R1-S44@4800M                                   greedy policy (the plain network name)
    E7-A4-R1-S44@4800M~policy(temperature=0.5)           sampled
    E7-A8-R1-S44@6800M~gumbel(sims=256,k=8,fpu=0.2)      Gumbel root search
    E7-A8-R1-S44@6800M~search(sims=128,determinize=true,node_filter=all,subsample=1,backend=cpp,fpu=0)
    HeuristicBot                                         a rule-based bot, no network

A search spec's id names every field, defaults included, so it never depends on what a default is
today. Only the greedy policy is abbreviated, to the bare network name, because that is the player
everyone means by a checkpoint's name.

Standard library only: tools/lib/leaderboard.py imports this, and it runs on CI without torch.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from typing import Dict, Final, Mapping, Optional, Tuple, Union


@dataclass(frozen=True)
class Policy:
    """The network's own policy: argmax at temperature <= 0.05 (the tournament's greedy cut-off),
    otherwise sampled at `temperature`."""

    temperature: float = 0.0


@dataclass(frozen=True)
class Gumbel:
    """A noise-free Gumbel root (ai/search/gumbel_root.py): the `k` most probable moves,
    sequential halving over `sims` network evaluations, first-play urgency reduction `fpu`."""

    sims: int = 256
    k: int = 8
    fpu: float = 0.2


@dataclass(frozen=True)
class Search:
    """Batched PUCT search (ai/search/batched_mcts.py). `node_filter` is "all" or "card"
    (SELECT_CARD / SELECT_PLAY_MODE only); `subsample` is the fraction of those searched."""

    sims: int = 64
    determinize: bool = False
    node_filter: str = "all"
    subsample: float = 1.0
    backend: str = "cpp"
    fpu: float = 0.0


@dataclass(frozen=True)
class Bot:
    """A rule-based bot with no network: `heuristic` or `random`."""

    name: str = "heuristic"


Inference = Union[Policy, Gumbel, Search, Bot]

KINDS: Final[Dict[str, type]] = {"policy": Policy, "gumbel": Gumbel, "search": Search, "bot": Bot}
_KIND_OF: Final[Dict[type, str]] = {cls: kind for kind, cls in KINDS.items()}
#: A bot's player id, and the load_agent spec it plays as.
BOTS: Final[Dict[str, Tuple[str, str]]] = {"heuristic": ("HeuristicBot", "heuristic"),
                                           "random": ("RandomBot", "random")}
_NODE_FILTERS: Final = ("all", "card")
_BACKENDS: Final = ("cpp", "python")
_ID_RE = re.compile(r"^(?P<network>[^~]+?)(?:~(?P<kind>[a-z]+)(?:\((?P<args>[^()]*)\))?)?$")


class PlayerSpecError(ValueError):
    pass


def kind_of(spec: Inference) -> str:
    return _KIND_OF[type(spec)]


def _fmt(value: Union[int, float, str, bool]) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _check(spec: Inference) -> None:
    """Refuse a spec the agent loader would not play as written."""
    if isinstance(spec, Policy) and spec.temperature < 0:
        raise PlayerSpecError(f"temperature must be >= 0, got {spec.temperature}")
    if isinstance(spec, (Gumbel, Search)) and spec.sims < 1:
        raise PlayerSpecError(f"sims must be >= 1, got {spec.sims}")
    if isinstance(spec, Gumbel) and spec.k < 1:
        raise PlayerSpecError(f"k must be >= 1, got {spec.k}")
    if isinstance(spec, Search):
        if spec.node_filter not in _NODE_FILTERS:
            raise PlayerSpecError(f"node_filter must be one of {_NODE_FILTERS}, got {spec.node_filter!r}")
        if spec.backend not in _BACKENDS:
            raise PlayerSpecError(f"backend must be one of {_BACKENDS}, got {spec.backend!r}")
        if not 0 < spec.subsample <= 1:
            raise PlayerSpecError(f"subsample must be in (0, 1], got {spec.subsample}")
    if isinstance(spec, Bot) and spec.name not in BOTS:
        raise PlayerSpecError(f"bot must be one of {sorted(BOTS)}, got {spec.name!r}")


def player_id(network: Optional[str], spec: Inference) -> str:
    """The canonical id of `spec` run on `network` (None for a bot)."""
    _check(spec)
    if isinstance(spec, Bot):
        if network is not None:
            raise PlayerSpecError("a bot has no network")
        return BOTS[spec.name][0]
    if not network:
        raise PlayerSpecError(f"a {kind_of(spec)} player needs a network")
    if "~" in network or ":" in network:
        raise PlayerSpecError(f"a network name may not contain '~' or ':': {network!r}")
    if spec == Policy():
        return network
    fields = ",".join(f"{f.name}={_fmt(getattr(spec, f.name))}" for f in dataclasses.fields(spec))
    return f"{network}~{kind_of(spec)}({fields})"


def _coerce(cls: type, name: str, raw: str) -> Union[int, float, str, bool]:
    default = getattr(cls(), name)
    if isinstance(default, bool):
        if raw not in ("true", "false"):
            raise PlayerSpecError(f"{name} must be true or false, got {raw!r}")
        return raw == "true"
    try:
        if isinstance(default, int):
            return int(raw)
        if isinstance(default, float):
            return float(raw)
    except ValueError:
        raise PlayerSpecError(f"{name}: not a number: {raw!r}") from None
    return raw


def spec_from_dict(d: Mapping[str, object]) -> Inference:
    """The spec stored in leaderboard/players.json: {"kind": ..., <field>: <value>, ...}. Every
    field is required, so a stored spec never silently takes up a changed default."""
    kind = d.get("kind")
    if not isinstance(kind, str) or kind not in KINDS:
        raise PlayerSpecError(f"unknown inference kind {kind!r}; one of {sorted(KINDS)}")
    cls = KINDS[kind]
    names = [f.name for f in dataclasses.fields(cls)]
    extra = sorted(set(d) - set(names) - {"kind"})
    missing = [n for n in names if n not in d]
    if extra or missing:
        raise PlayerSpecError(f"{kind} spec: unknown fields {extra}, missing fields {missing}")
    values: Dict[str, Union[int, float, str, bool]] = {}
    for n in names:
        v = d[n]
        default = getattr(cls(), n)
        if isinstance(default, bool) != isinstance(v, bool) or (
                isinstance(default, (int, float)) and not isinstance(v, (int, float))) or (
                isinstance(default, str) and not isinstance(v, str)):
            raise PlayerSpecError(f"{kind}.{n}: expected {type(default).__name__}, got {v!r}")
        if isinstance(v, bool) or isinstance(v, str):
            values[n] = v
        elif isinstance(v, (int, float)):
            values[n] = float(v) if isinstance(default, float) else v
    spec = cls(**values)
    _check(spec)
    return spec


def spec_to_dict(spec: Inference) -> Dict[str, object]:
    return {"kind": kind_of(spec), **dataclasses.asdict(spec)}


def parse_player_id(pid: str) -> Tuple[Optional[str], Inference]:
    """(network, spec) of a player id; the inverse of `player_id`. A search spec's fields may be
    given in any order or left out (taking today's defaults) -- the canonical id spells them all,
    which is what `player_id` of the result returns."""
    for name, (bot_id, _) in BOTS.items():
        if pid == bot_id:
            return None, Bot(name)
    m = _ID_RE.match(pid.strip())
    if not m:
        raise PlayerSpecError(f"not a player id: {pid!r}")
    network, kind, args = m.group("network"), m.group("kind"), m.group("args")
    if kind is None:
        return network, Policy()
    if kind not in KINDS or kind == "bot":
        raise PlayerSpecError(f"unknown inference kind {kind!r} in {pid!r}")
    cls = KINDS[kind]
    names = {f.name for f in dataclasses.fields(cls)}
    values: Dict[str, Union[int, float, str, bool]] = {}
    for part in (args or "").split(","):
        if not part.strip():
            continue
        key, sep, raw = part.partition("=")
        key = key.strip()
        if not sep or key not in names:
            raise PlayerSpecError(f"{kind}: unknown field {key!r} in {pid!r}; fields {sorted(names)}")
        values[key] = _coerce(cls, key, raw.strip())
    spec = cls(**values)
    _check(spec)
    return network, spec


def load_spec(spec: Inference, checkpoint: Optional[str]) -> str:
    """The `tools.lib.player_agent.load_agent` string that plays `spec` on `checkpoint`. A policy
    pins its own temperature, so the tournament's shared --temperature cannot change it."""
    _check(spec)
    if isinstance(spec, Bot):
        return BOTS[spec.name][1]
    if not checkpoint:
        raise PlayerSpecError(f"a {kind_of(spec)} player needs a checkpoint file")
    if ":" in checkpoint:
        raise PlayerSpecError(f"agent specs are ':'-separated, so a checkpoint path may not contain one: {checkpoint!r}")
    if isinstance(spec, Policy):
        return f"temp:{_fmt(spec.temperature)}:{checkpoint}"
    if isinstance(spec, Gumbel):
        return f"gumbel:{checkpoint}:{spec.sims}:{spec.k}:{_fmt(spec.fpu)}"
    return (f"search:{checkpoint}:{spec.sims}:{'determinize' if spec.determinize else 'none'}:"
            f"{spec.node_filter}:{_fmt(spec.subsample)}:{spec.backend}:{_fmt(spec.fpu)}")
