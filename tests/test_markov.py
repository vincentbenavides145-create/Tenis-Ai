from __future__ import annotations

import pytest

from tenis_ai.markov import (
    hold_probability,
    match_distribution,
    match_win_probability,
    set_outcomes,
    tiebreak_win,
)


def test_hold_probability_known_values():
    assert hold_probability(0.5) == pytest.approx(0.5)
    assert hold_probability(0.6) == pytest.approx(0.7357, abs=1e-4)
    assert hold_probability(0.7) > hold_probability(0.6)


def test_tiebreak_symmetric_and_monotonic():
    assert tiebreak_win(0.6, 0.6, 7, True) == pytest.approx(0.5)
    assert tiebreak_win(0.6, 0.6, 10, False) == pytest.approx(0.5)
    assert tiebreak_win(0.66, 0.6, 7, True) > 0.5


def test_set_distribution_sums_to_one_and_valid_scores():
    outcomes = set_outcomes(0.64, 0.6, True, 7)
    assert sum(p for *_, p in outcomes) == pytest.approx(1.0)
    for ga, gb, ba, bb, _p in outcomes:
        assert (max(ga, gb) == 6 and abs(ga - gb) >= 2) or max(ga, gb) == 7
        assert ba >= 0 and bb >= 0


@pytest.mark.parametrize("best_of", [3, 5])
def test_match_distribution_consistent(best_of):
    d = match_distribution(0.66, 0.62, best_of)
    assert d.p_win_a == pytest.approx(match_win_probability(0.66, 0.62, best_of))
    for dist in (d.total_games, d.games_a, d.total_breaks, d.total_sets, d.first_set_games):
        assert sum(dist.values()) == pytest.approx(1.0)
    assert sum(d.set_scores.values()) == pytest.approx(1.0)
    assert 0 < d.p_tiebreak < 1
    assert match_win_probability(0.65, 0.65, best_of) == pytest.approx(0.5)


def test_best_of_five_favours_favourite():
    assert match_win_probability(0.68, 0.63, 5) > match_win_probability(0.68, 0.63, 3) > 0.5
