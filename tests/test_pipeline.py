from __future__ import annotations

import random

import pytest

from tenis_ai.backtesting import run_backtest, winner_objective
from tenis_ai.data.api_tennis import ApiTennisClient
from tenis_ai.data.sackmann import Match, SetScore, parse_score
from tenis_ai.markets import actual_outcomes, build_candidates, settle
from tenis_ai.model import predict_match
from tenis_ai.params import DEFAULT_PARAMS, TennisParams, fit_params, load_params, save_params
from tenis_ai.players import PlayerNotFound, resolve_player
from tenis_ai.ratings import RatingSystem
from tenis_ai.selection import select_markets, top_pick


def _match(i, w, lname, wname, l, date, score="6-4 6-3", surface="Hard", stats=True):
    sets, completed = parse_score(score)
    st = {}
    if stats:
        st = {"w_svpt": 60, "w_1stWon": 30, "w_2ndWon": 12, "w_bpSaved": 2, "w_bpFaced": 3,
              "l_svpt": 62, "l_1stWon": 25, "l_2ndWon": 10, "l_bpSaved": 3, "l_bpFaced": 6}
    return Match(f"t{i}", "Torneo", surface, "A", date, "R32", i, 3, w, l, wname, lname, score, sets, completed, st)


def _league(n=300, seed=1):
    rng = random.Random(seed)
    strength = {p: rng.gauss(0, 1) for p in range(1, 13)}
    names = {p: f"Jugador {chr(64 + p)}lfa Apellido{chr(64 + p)}x" for p in strength}
    out = []
    for i in range(n):
        a, b = rng.sample(sorted(strength), 2)
        pa = 1 / (1 + 10 ** ((strength[b] - strength[a]) / 1.5))
        w, l = (a, b) if rng.random() < pa else (b, a)
        date = f"20{18 + i // 100}{(i % 100) // 9 + 1:02d}{(i % 9) + 1:02d}"
        score = rng.choice(["6-4 6-3", "7-6(4) 6-4", "6-3 4-6 6-2", "6-7(5) 7-5 6-4"])
        out.append(_match(i, w, names[l], names[w], l, date, score))
    out.sort(key=lambda m: m.order_key)
    return out


def test_parse_score_variants():
    assert parse_score("6-3 3-6 7-6(5)") == ((SetScore(6, 3), SetScore(3, 6), SetScore(7, 6)), True)
    assert parse_score("6-4 2-1 RET")[1] is False
    assert parse_score("W/O") == ((), False)
    assert parse_score("2-6 6-4 [11-9]")[1] is False


def test_ratings_learn_and_resolve_names():
    matches = _league()
    ratings = RatingSystem(DEFAULT_PARAMS)
    for m in matches:
        ratings.update(m)
    assert resolve_player(ratings, "Jugador Clfa ApellidoCx") == 3
    assert resolve_player(ratings, "apellidocx") == 3
    assert resolve_player(ratings, "J. ApellidoCx") == 3
    with pytest.raises(PlayerNotFound):
        resolve_player(ratings, "Nadie Desconocido")
    pa, pb = ratings.serve_probabilities(1, 2, "Hard")
    assert 0.35 <= pa <= 0.9 and 0.35 <= pb <= 0.9


def test_prediction_markets_and_settlement():
    matches = _league()
    ratings = RatingSystem(DEFAULT_PARAMS)
    for m in matches:
        ratings.update(m)
    pred = predict_match(ratings, 1, 2, "Hard", 3)
    cands = build_candidates(pred)
    assert any(c["group"] == "winner" for c in cands)
    by_id = {c["id"]: c for c in cands}
    assert by_id["winner_a"]["probability"] + by_id["winner_b"]["probability"] == pytest.approx(1.0, abs=1e-3)
    assert not any("andicap" in c["market"] for c in cands)

    m = _match(999, 1, "B", "A", 2, "20300101", "7-6(3) 4-6 6-4")
    outcome = actual_outcomes(m, a_is_winner=True)
    assert outcome["total_games"] == 7 + 6 + 4 + 6 + 6 + 4
    assert outcome["tiebreak"] is True and outcome["set_score"] == (2, 1)
    assert outcome["breaks_a"] == 3 and outcome["breaks_b"] == 1
    assert settle(by_id["games_over_21.5"], outcome) is True
    assert settle(by_id["winner_b"], outcome) is False
    assert actual_outcomes(_match(1000, 1, "B", "A", 2, "20300101", "6-4 2-1 RET"), True) is None


def test_selection_rules_and_expected_value():
    cands = [
        {"id": "x1", "category": "Juegos", "group": "g", "probability": 0.82, "selection": "a", "explanation": ""},
        {"id": "x2", "category": "Juegos", "group": "g", "probability": 0.70, "selection": "b", "explanation": ""},
        {"id": "x3", "category": "Juegos", "group": "h", "probability": 0.95, "selection": "c", "explanation": ""},
        {"id": "x4", "category": "Juegos", "group": "i", "probability": 0.55, "selection": "d", "explanation": ""},
    ]
    sel = select_markets(cands)
    assert [s["id"] for s in sel] == ["x1"]  # una por grupo, fuera de rango excluidas
    priced = select_markets(cands, odds={"x1": 1.10, "x2": 1.60})
    assert [s["id"] for s in priced] == ["x2"]  # x1 tiene EV negativo; x2 EV +12 %
    assert priced[0]["edge"] == pytest.approx(0.12)
    assert top_pick(priced)["id"] == "x2"


def test_backtest_has_no_lookahead_and_fit_runs():
    matches = _league(400)
    result = run_backtest(matches, DEFAULT_PARAMS, "20200101", "20211231", min_matches=5)
    assert result.evaluated > 0
    assert result.summary()["winner"]["brier"] < 0.3
    obj = winner_objective(matches, "20200101", "20211231", min_matches=5)
    fitted = fit_params(obj, grid={"elo_weight": [0.3, 1.0]}, passes=1)
    assert fitted.fitted and fitted.log_loss <= fitted.baseline_log_loss


def test_params_roundtrip(tmp_path):
    path = tmp_path / "p.json"
    save_params(TennisParams(elo_weight=0.42, fitted=True), path)
    assert load_params(path).elo_weight == 0.42
    assert load_params(tmp_path / "missing.json") == DEFAULT_PARAMS


def test_api_tennis_parsing(monkeypatch):
    payload = {
        "success": 1,
        "result": [
            {"event_key": 1, "event_date": "2026-11-16", "event_time": "14:00", "event_first_player": "C. Alcaraz",
             "event_second_player": "J. Sinner", "tournament_name": "ATP Finals", "tournament_round": "Final",
             "event_type_type": "Atp Singles", "event_status": "Finished", "event_winner": "Second Player",
             "scores": [{"score_first": "4", "score_second": "6", "score_set": "1"},
                        {"score_first": "6", "score_second": "7", "score_set": "2"}]},
            {"event_key": 2, "event_type_type": "Atp Doubles", "event_status": "Finished"},
        ],
    }

    class Resp:
        status_code = 200

        def json(self):
            return payload

    monkeypatch.setattr("tenis_ai.data.api_tennis.requests.get", lambda *a, **k: Resp())
    client = ApiTennisClient(api_key="x")
    fixtures = client.atp_singles_fixtures("2026-11-16")
    assert len(fixtures) == 1 and fixtures[0].player_a == "C. Alcaraz"
    results = client.finished_atp_results("2026-11-16", "2026-11-16")
    assert results[0]["winner"] == "J. Sinner" and results[0]["sets"] == [(6, 4), (7, 6)]
