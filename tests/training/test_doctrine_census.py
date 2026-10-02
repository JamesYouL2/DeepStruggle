"""The doctrine census: plays are logged with the right card, and the rules read them correctly."""
import numpy as np

from ai.eval import doctrine_census as D


def _random_policy(seed: int):
    rng = np.random.default_rng(seed)

    def act(obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        return np.array([rng.choice(np.flatnonzero(m)) if m.any() else 0 for m in masks], dtype=np.int32)
    return act


def test_collect_logs_plays_headlines_and_coups() -> None:
    d = D.collect(_random_policy(0), 6, seed=3, envs=4)
    assert d["games"] == 6
    assert d["plays"] and d["headlines"]
    info = D.cards()
    for r in d["plays"]:
        assert r["card"] in info and r["mode"] in r["legal"] and len(r["legal"]) >= 2
        assert r["fires"] == (r["mode"] == "event" or (r["owner"] == "opp" and r["mode"] != "space"))
    for h in d["headlines"]:
        assert h["card"] in h["hand"]
    assert sum(1 for h in d["headlines"] if h["turn"] == 1) == 12      # both sides, every game


def test_rules_read_mode_and_side() -> None:
    def play(card: int, side: str, mode: str, owner: str = "own", turn: int = 3, vp: int = 0) -> dict:
        legal = ["event", "space", "influence", "coup"]
        return {"card": card, "side": side, "mode": mode, "owner": owner, "turn": turn, "vp": vp, "ihc": False,
                "legal": legal, "fires": mode == "event" or (owner == "opp" and mode != "space")}
    by = {name: (ap, co) for name, ap, co in D.rules()}
    ap, co = by["US always events The Voice of America"]
    assert ap(play(D.VOICE_OF_AMERICA, "US", "event")) and co(play(D.VOICE_OF_AMERICA, "US", "event"))
    assert not co(play(D.VOICE_OF_AMERICA, "US", "influence"))
    assert not ap(play(D.VOICE_OF_AMERICA, "USSR", "event", owner="opp"))
    ap, co = by["USSR never lets The Voice of America fire"]
    assert co(play(D.VOICE_OF_AMERICA, "USSR", "space", owner="opp"))
    assert not co(play(D.VOICE_OF_AMERICA, "USSR", "influence", owner="opp"))
    ap, _ = by["USSR always events De-Stalinization before turn 5"]
    assert ap(play(D.DESTALINIZATION, "USSR", "event", turn=4)) and not ap(play(D.DESTALINIZATION, "USSR", "event", turn=5))


def test_card_ids_match_their_names() -> None:
    info = D.cards()
    want = {D.GRAIN_SALES: "Grain Sales to Soviets", D.VOICE_OF_AMERICA: "The Voice of America",
            D.COLONIAL_REAR_GUARDS: "Colonial Rear Guards", D.ALDRICH_AMES: "Aldrich Ames Remix",
            D.DESTALINIZATION: "De-Stalinization", D.DECOLONIZATION: "Decolonization",
            D.NAZI_SCIENTIST: "Captured Nazi Scientist", D.JUNTA: "Junta", D.TERRORISM: "Terrorism",
            D.FORMOSAN: "Formosan Resolution", D.COMECON: "Comecon", D.NORAD: "NORAD", D.STAR_WARS: "Star Wars",
            D.OPEC: "OPEC", D.ALLIANCE_FOR_PROGRESS: "Alliance for Progress", D.CHE: "Che"}
    assert {c: info[c]["name"] for c in want} == want
    names = {r["name"] for r in D._map()}
    assert set(D.OPEC_COUNTRIES) <= names
