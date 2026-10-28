import json
import os
import time
from typing import Dict, List, Optional, Union, cast
from web.server.replay_types import (
    ReplayListErrorDict,
    ReplayLogDict,
    ReplayStepDict,
    ReplaySummaryDict,
    ReplayMetadataDict,
    ReplayResultDict,
    GameStateDict,
    ReplayActionDict,
    ReplayPolicyDict,
    ReplayCriticDict,
    ReplayTraceMetaDict,
)

_ROOT_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_REPLAYS_DIR: str = os.path.join(_ROOT_DIR, "data", "replays")

#: Environment variable that overrides where replays are read from and written to.
REPLAYS_DIR_ENV: str = "TS_REPLAYS_DIR"

#: The only files the replay directory is read for.
REPLAY_SUFFIX: str = ".tslog.json"


def replays_dir() -> str:
    """Where replays live, resolved per call rather than fixed at import.

    `data/replays/` is git-ignored, so tests cannot rely on it holding anything -- and a test
    that skips itself when it is empty is worse than no test, because it passes on every machine
    while checking nothing. Resolving the location per call lets a fixture point the server at a
    directory it has just generated a replay into, so the schema tests always run against real
    output of the current writer. It also lets two servers run against separate directories.
    """
    return os.environ.get(REPLAYS_DIR_ENV) or DEFAULT_REPLAYS_DIR


class ReplayLogger:
    def __init__(self, game_id: str, seed: int, us_player: str = "US Player", ussr_player: str = "USSR Player") -> None:
        self.game_id: str = game_id
        self.seed: int = seed
        self.us_player: str = us_player
        self.ussr_player: str = ussr_player
        self.created_at: str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.steps: List[ReplayStepDict] = []
        self.result: Optional[ReplayResultDict] = None
        self.trace: Optional[ReplayTraceMetaDict] = None
        #: Set when the game began from a loaded position (a workbench link) rather than from
        #: the seed; the token is a `pos=` link's (web/ui/src/game/position.ts). Such a replay cannot
        #: be re-driven from the seed, and says so by carrying this.
        self.start_position: Optional[str] = None

    def log_step(
        self,
        step_index: int,
        turn: int,
        ar: int,
        phase: str,
        player: str,
        action: ReplayActionDict,
        description: str,
        state_snapshot: GameStateDict,
        policy: Optional[ReplayPolicyDict] = None,
        critic: Optional[ReplayCriticDict] = None,
    ) -> None:
        step: ReplayStepDict = {
            "step_index": step_index,
            "turn": turn,
            "ar": ar,
            "phase": phase,
            "player": player,
            "action": action,
            "description": description,
            "state_snapshot": state_snapshot,
        }
        # Written only when there is something to write: a heuristic bot has no distribution and
        # a replay without a trace must stay byte-identical to what this writer produced before.
        if policy is not None:
            step["policy"] = policy
        if critic is not None:
            step["critic"] = critic
        self.steps.append(step)

    def set_trace_meta(self, trace: ReplayTraceMetaDict) -> None:
        """Record which model, build and settings produced the per-step trace."""
        self.trace = trace

    def set_result(self, winner: str, margin: int, end_turn: int, reason: str) -> None:
        self.result = {
            "winner": winner,
            "margin": margin,
            "end_turn": end_turn,
            "reason": reason,
        }

    def to_dict(self) -> ReplayLogDict:
        metadata: ReplayMetadataDict = {
            "game_id": self.game_id,
            "created_at": self.created_at,
            "seed": self.seed,
            "players": {
                "US": self.us_player,
                "USSR": self.ussr_player,
            },
            "total_steps": len(self.steps),
            "result": self.result,
        }
        if self.trace is not None:
            metadata["trace"] = self.trace
        if self.start_position is not None:
            metadata["start_position"] = self.start_position
        return {
            "version": "1.0",
            "metadata": metadata,
            "initial_state": {
                "seed": self.seed,
            },
            "steps": self.steps,
        }

    def save(self, filepath: Optional[str] = None) -> str:
        target = replays_dir()
        os.makedirs(target, exist_ok=True)
        if not filepath:
            filepath = os.path.join(target, f"{self.game_id}.tslog.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return filepath


class ReplayManager:
    """The replay directory as the local server serves it: `*.tslog.json` files only, nothing
    outside the directory."""

    @staticmethod
    def list_replays() -> List[Union[ReplaySummaryDict, ReplayListErrorDict]]:
        """Every replay in the directory, newest first; a file that cannot be read is listed after
        them with the reason, rather than left out."""
        target = replays_dir()
        os.makedirs(target, exist_ok=True)
        replays: List[ReplaySummaryDict] = []
        broken: List[ReplayListErrorDict] = []
        for fname in sorted(os.listdir(target)):
            if not fname.endswith(REPLAY_SUFFIX):
                continue
            try:
                with open(os.path.join(target, fname), "r", encoding="utf-8") as f:
                    data: object = json.load(f)
                replays.append(_summary(fname, data))
            except (OSError, ValueError, TypeError) as e:
                # ValueError: not JSON (json.JSONDecodeError), not UTF-8 (UnicodeDecodeError), or
                # fields that are not what a replay holds (_summary).
                broken.append({"filename": fname, "error": f"{type(e).__name__}: {e}"})
        replays.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return [*replays, *broken]

    @staticmethod
    def load_replay(filename: str) -> Optional[ReplayLogDict]:
        """The replay `filename` (with or without its `.tslog.json`), or None when the directory
        has no such replay. Raises ValueError for a file there that is not a replay."""
        target = os.path.realpath(replays_dir())
        name = filename if filename.endswith(REPLAY_SUFFIX) else f"{filename}{REPLAY_SUFFIX}"
        full_path = os.path.join(target, name)
        # Served over HTTP by the local workbench: nothing outside the replay directory.
        if not os.path.realpath(full_path).startswith(target + os.sep) or not os.path.isfile(full_path):
            return None
        with open(full_path, "r", encoding="utf-8") as f:
            data: object = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"{name} is not a replay: a JSON {type(data).__name__}, not an object")
        return cast(ReplayLogDict, data)


def _summary(fname: str, data: object) -> ReplaySummaryDict:
    """What the replay picker shows of one replay. Raises ValueError (or TypeError) for anything
    not shaped like one."""
    if not isinstance(data, dict):
        raise ValueError(f"a JSON {type(data).__name__}, not an object")
    meta = data.get("metadata")
    if isinstance(meta, dict):
        steps = data.get("steps", [])
        return {
            "filename": fname,
            "game_id": str(meta.get("game_id", fname)),
            "created_at": str(meta.get("created_at", "")),
            "players": _players(meta.get("players", {})),
            "total_steps": int(meta.get("total_steps", len(steps) if isinstance(steps, list) else 0)),
            "result": _result(meta.get("result")),
        }
    # Older flat files: the steps under "steps", "events" or "logs", the outcome at the top level.
    steps_list = data.get("steps") or data.get("events") or data.get("logs") or []
    fallback: Optional[ReplayResultDict] = None
    if "winner" in data:
        fallback = {
            "winner": str(data["winner"]),
            "margin": int(data.get("final_vp", 0)),
            "end_turn": int(data.get("final_turn", 1)),
            "reason": "Terminal Evaluation",
        }
    return {
        "filename": fname,
        "game_id": str(data.get("game_id", fname)),
        "created_at": str(data.get("created_at", "")),
        "players": _players(data.get("players", {})),
        "total_steps": int(data.get("total_steps", len(steps_list) if isinstance(steps_list, list) else 0)),
        "result": fallback,
    }


def _players(v: object) -> Dict[str, str]:
    if not isinstance(v, dict):
        raise ValueError("players is not an object")
    return {str(k): str(p) for k, p in v.items()}


def _result(v: object) -> Optional[ReplayResultDict]:
    if v is None:
        return None
    if not isinstance(v, dict):
        raise ValueError("result is not an object")
    out: ReplayResultDict = {}
    if "winner" in v:
        out["winner"] = str(v["winner"])
    if "margin" in v:
        out["margin"] = int(v["margin"])
    if "end_turn" in v:
        out["end_turn"] = int(v["end_turn"])
    if "reason" in v:
        out["reason"] = str(v["reason"])
    return out
