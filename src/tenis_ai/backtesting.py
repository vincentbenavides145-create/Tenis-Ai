"""Backtest walk-forward: cada partido se predice solo con partidos anteriores.

Se miden dos conjuntos (igual que en Deport-AI):
* ``all``: todos los mercados calculados (calibración global).
* ``shown``: solo lo que pasa el filtro de presentación (60–88 %, una por grupo,
  máx. 4 por categoría), es decir, lo que ve el usuario.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .data.sackmann import Match
from .evaluation import brier_score, calibration_bins, log_loss
from .markets import actual_outcomes, build_candidates, settle
from .model import blended_probability, predict_match
from .params import TennisParams
from .ratings import RatingSystem
from .selection import select_markets

EVAL_LEVELS = {"G", "M", "A", "F"}
SHOWN_BINS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.88]


def _eligible(match: Match, ratings: RatingSystem, min_matches: int) -> bool:
    if match.level not in EVAL_LEVELS or not match.completed:
        return False
    return (
        ratings.players[match.winner_id].matches >= min_matches
        and ratings.players[match.loser_id].matches >= min_matches
    )


def _a_is_winner(match: Match) -> bool:
    # A = id menor: asignación determinista que no depende del resultado.
    return match.winner_id < match.loser_id


def winner_objective(
    matches: list[Match],
    eval_from: str,
    eval_to: str,
    min_matches: int = 10,
) -> Callable[[TennisParams], tuple[float, int]]:
    """Log-loss del ganador, walk-forward, para el ajuste de parámetros."""

    def objective(params: TennisParams) -> tuple[float, int]:
        ratings = RatingSystem(params)
        total, count = 0.0, 0
        for match in matches:
            if match.date > eval_to:
                break
            if match.date >= eval_from and _eligible(match, ratings, min_matches):
                p_winner, *_ = blended_probability(
                    ratings, match.winner_id, match.loser_id, match.surface, match.best_of
                )
                total += -math.log(max(1e-12, p_winner))
                count += 1
            ratings.update(match)
        return (total / count if count else float("inf")), count

    return objective


def games_objective(
    matches: list[Match],
    params: TennisParams,
    eval_from: str,
    eval_to: str,
    min_matches: int = 10,
    sample_every: int = 4,
) -> Callable[[float], tuple[float, int]]:
    """Log-loss del total exacto de juegos en función de form_sd (muestra 1 de cada N)."""
    from .model import _align_serves, distribution_with_form

    ratings = RatingSystem(params)
    cases: list[tuple[float, float, float, int, int]] = []
    seen = 0
    for match in matches:
        if match.date > eval_to:
            break
        if match.date >= eval_from and _eligible(match, ratings, min_matches):
            outcome = actual_outcomes(match, True)
            if outcome and outcome["games_supported"]:
                seen += 1
                if seen % sample_every == 0:
                    p, _pe, _ps, pa, pb = blended_probability(
                        ratings, match.winner_id, match.loser_id, match.surface, match.best_of
                    )
                    cases.append((pa, pb, p, match.best_of, outcome["total_games"]))
        ratings.update(match)

    def objective(sd: float) -> tuple[float, int]:
        total = 0.0
        for pa, pb, p, bo, games in cases:
            sa, sb = _align_serves(pa, pb, p, bo, sd)
            dist = distribution_with_form(sa, sb, bo, sd)
            total += -math.log(max(1e-9, dist.total_games.get(games, 0.0)))
        return (total / len(cases) if cases else float("inf")), len(cases)

    return objective


@dataclass
class BacktestResult:
    evaluated: int = 0
    winner_probs: list[float] = field(default_factory=list)  # prob. asignada al jugador A
    winner_outcomes: list[int] = field(default_factory=list)
    records: list[dict[str, Any]] = field(default_factory=list)
    elo_only: list[float] = field(default_factory=list)
    sr_only: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {"evaluated_matches": self.evaluated}
        if self.winner_probs:
            o = self.winner_outcomes
            fav_hits = sum(1 for p, y in zip(self.winner_probs, o) if (p >= 0.5) == bool(y))
            out["winner"] = {
                "brier": round(brier_score(self.winner_probs, o), 4),
                "log_loss": round(log_loss(self.winner_probs, o), 4),
                "accuracy": round(fav_hits / len(o), 4),
                "brier_elo_only": round(brier_score(self.elo_only, o), 4),
                "brier_serve_return_only": round(brier_score(self.sr_only, o), 4),
            }
        out["shown"] = summarize([r for r in self.records if r["shown"]], SHOWN_BINS)
        out["all"] = summarize(self.records, [i / 10 for i in range(11)])
        return out


def run_backtest(
    matches: list[Match],
    params: TennisParams,
    eval_from: str,
    eval_to: str,
    min_matches: int = 10,
    full_markets: bool = True,
    progress: Callable[[int], None] | None = None,
) -> BacktestResult:
    ratings = RatingSystem(params)
    result = BacktestResult()
    for match in matches:
        if match.date > eval_to:
            break
        if match.date >= eval_from and _eligible(match, ratings, min_matches):
            a_win = _a_is_winner(match)
            a, b = (match.winner_id, match.loser_id) if a_win else (match.loser_id, match.winner_id)
            pred = predict_match(ratings, a, b, match.surface, match.best_of)
            result.evaluated += 1
            result.winner_probs.append(pred.p_win_a)
            result.winner_outcomes.append(1 if a_win else 0)
            result.elo_only.append(pred.p_elo)
            result.sr_only.append(pred.p_serve_return)
            if full_markets:
                outcome = actual_outcomes(match, a_win)
                if outcome is not None:
                    candidates = build_candidates(pred)
                    shown_ids = {c["id"] for c in select_markets(candidates)}
                    for cand in candidates:
                        won = settle(cand, outcome)
                        if won is None:
                            continue
                        result.records.append(
                            {
                                "category": cand["category"],
                                "group": cand["group"],
                                "probability": cand["probability"],
                                "won": won,
                                "shown": cand["id"] in shown_ids,
                            }
                        )
            if progress and result.evaluated % 500 == 0:
                progress(result.evaluated)
        ratings.update(match)
    return result


def summarize(records: list[dict[str, Any]], bins: Iterable[float]) -> dict[str, Any]:
    if not records:
        return {"count": 0}

    def block(items: list[dict[str, Any]]) -> dict[str, Any]:
        p = [r["probability"] for r in items]
        o = [1 if r["won"] else 0 for r in items]
        return {
            "count": len(items),
            "mean_predicted": round(sum(p) / len(p), 4),
            "hit_rate": round(sum(o) / len(o), 4),
            "gap": round(sum(o) / len(o) - sum(p) / len(p), 4),
            "brier": round(brier_score(p, o), 4),
        }

    probs = [r["probability"] for r in records]
    outcomes = [1 if r["won"] else 0 for r in records]
    by_cat: dict[str, list] = defaultdict(list)
    by_group: dict[str, list] = defaultdict(list)
    for r in records:
        by_cat[r["category"]].append(r)
        by_group[r["group"]].append(r)
    return {
        **block(records),
        "log_loss": round(log_loss(probs, outcomes), 4),
        "calibration": [
            {
                "bin": f"{b.lower * 100:.0f}-{b.upper * 100:.0f}%",
                "count": b.count,
                "predicted": round(b.mean_predicted, 4),
                "observed": round(b.observed_frequency, 4),
            }
            for b in calibration_bins(probs, outcomes, bins=list(bins))
            if b.count
        ],
        "by_category": {k: block(v) for k, v in sorted(by_cat.items())},
        "by_group": {k: block(v) for k, v in sorted(by_group.items())},
    }
