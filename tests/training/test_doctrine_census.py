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
            D.OPEC: "OPEC", D.ALLIANCE_FOR_PROGRESS: "Alliance for Progress", D.CHE: "Che", D.FIVE_YEAR_PLAN: "Five Year Plan"}
    assert {c: info[c]["name"] for c in want} == want
    names = {r["name"] for r in D._map()}
    assert set(D.OPEC_COUNTRIES) <= names


def test_last_round() -> None:
    assert D.is_last_round(3, 6) and not D.is_last_round(4, 6) and D.is_last_round(4, 7) and D.is_last_round(9, 8)


def test_discard_cards_log_the_rest_of_the_hand() -> None:
    d = D.collect(_random_policy(1), 30, seed=5, envs=8)
    rows = [r for r in d["plays"] if r["card"] in (D.FIVE_YEAR_PLAN, D.ALDRICH_AMES)]
    assert rows and all(0 <= r["other_scoring"] <= r["hand_other"] for r in rows)
    assert {c: D.cards()[c]["name"].endswith("Scoring") for c in D.SCORING_CARDS} == dict.fromkeys(D.SCORING_CARDS, True)


def test_discard_timing_complies_with_one_or_zero_other_cards() -> None:
    def play(hand_other: int, ar: int) -> dict:
        return {"card": D.FIVE_YEAR_PLAN, "side": "USSR", "mode": "event", "owner": "opp", "turn": 5, "ar": ar,
                "vp": 0, "ihc": False, "legal": ["event", "space", "influence"], "fires": True,
                "hand_other": hand_other, "other_scoring": 0}
    plays = [play(0, 7), play(1, 7), play(1, 6), play(2, 6), play(4, 2)]
    heads = [{"card": D.FIVE_YEAR_PLAN, "side": "USSR", "turn": 5, "hand": [D.FIVE_YEAR_PLAN]}]
    _md, s = D.report({"plays": plays, "headlines": heads, "coups": [], "boards": [], "games": 1}, {})
    r = s["rules"]["USSR plays Five Year Plan with one or zero other cards in hand"]
    assert r["n"] == 6 and r["complied"] == 3            # 0, 1 and 1 other card; a headline never complies
    assert r["last_round"] == 2 and r["hand_other"] == [1, 2, 2]
