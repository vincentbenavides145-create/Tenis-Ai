from __future__ import annotations

from tenis_ai.engine import Report
from tenis_ai.history import HistoryStore, outcome_from_api
from tenis_ai.markets import build_candidates
from tenis_ai.model import predict_match
from tenis_ai.params import DEFAULT_PARAMS
from tenis_ai.ratings import RatingSystem
from tenis_ai.selection import select_markets, top_pick
from tests.test_pipeline import _league


def _report() -> Report:
    ratings = RatingSystem(DEFAULT_PARAMS)
    for m in _league():
        ratings.update(m)
    pred = predict_match(ratings, 1, 2, "Hard", 3, name_a="Carlos Alcaraz", name_b="Jannik Sinner")
    selected = select_markets(build_candidates(pred))
    return Report(pred, selected, top_pick(selected), "")


class FakeClient:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def raw_fixtures(self, start, stop):
        self.calls.append((start, stop))
        return self.rows


def _row(status="Finished", winner="Second Player", scores=((6, 7), (6, 4), (4, 6)), key="99"):
    return {
        "event_key": key, "event_type_type": "Atp Singles", "event_status": status, "event_winner": winner,
        "event_first_player": "C. Alcaraz", "event_second_player": "J. Sinner",
        "scores": [{"score_first": str(a), "score_second": str(b), "score_set": str(i + 1)} for i, (a, b) in enumerate(scores)],
    }


def test_save_dedup_and_settle(tmp_path):
    store = HistoryStore(tmp_path / "h.jsonl")
    report = _report()
    store.save_prediction(report, "2026-11-17", "Finals - Turin", event_key="99", source="today")
    store.save_prediction(report, "2026-11-17", "Finals - Turin", event_key="99", source="today")
    records = store.load()
    assert len(records) == 1 and records[0]["status"] == "pending"
    assert records[0]["selections"], "debe guardar las selecciones mostradas"

    summary = store.settle_pending(FakeClient([_row()]), today="2026-11-18")
    assert summary["settled"] == 1
    rec = store.load()[0]
    assert rec["status"] == "settled" and rec["result"]["winner"] == "b"
    assert rec["result"]["total_games"] == 6 + 7 + 6 + 4 + 4 + 6
    assert all(s["won"] in (True, False, None) for s in rec["selections"])
    # Una vez liquidado no se sobrescribe aunque se vuelva a predecir.
    store.save_prediction(report, "2026-11-17", "Finals - Turin", event_key="99")
    assert store.load()[0]["status"] == "settled"
    assert "Histórico de predicciones" in store.report()


def test_pending_today_not_settled_and_void(tmp_path):
    store = HistoryStore(tmp_path / "h.jsonl")
    store.save_prediction(_report(), "2026-11-17", event_key="99")
    assert store.settle_pending(FakeClient([_row()]), today="2026-11-17")["pending"] == 0
    store.settle_pending(FakeClient([_row(status="Retired", winner="First Player")]), today="2026-11-18")
    assert store.load()[0]["status"] == "void"


def test_match_by_names_without_event_key_and_orientation():
    record = {"player_a": {"name": "Jannik Sinner"}, "player_b": {"name": "Carlos Alcaraz"}}
    outcome = outcome_from_api(_row(), record)
    # A = Sinner (segundo jugador en la API), que ganó 7-6 4-6 6-4
    assert outcome["winner"] == "a" and outcome["set_score"] == (2, 1)
    assert outcome["first_set_tiebreak"] is True and outcome["games_a"] == 7 + 4 + 6
    assert outcome_from_api(_row(status="Set 2"), record) is None
