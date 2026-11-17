"""Combina Elo y saque/resto en una predicción coherente para todos los mercados."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from .markov import MatchDistribution, match_distribution, match_win_probability
from .ratings import RatingSystem, inv_logit, logit

# Cuadratura de Gauss-Hermite (5 nodos) para integrar la "forma del día" ~ N(0, sd).
_GH = [(-2.0201828705, 0.0199532421), (-0.9585724646, 0.3936193232), (0.0, 0.9453087205),
       (0.9585724646, 0.3936193232), (2.0201828705, 0.0199532421)]
_NODES = [(x * math.sqrt(2.0), w / math.sqrt(math.pi)) for x, w in _GH]


@dataclass
class MatchPrediction:
    player_a: int
    player_b: int
    name_a: str
    name_b: str
    surface: str
    best_of: int
    p_elo: float
    p_serve_return: float
    p_win_a: float
    serve_a: float  # prob. media de punto al saque (tras el ajuste de coherencia)
    serve_b: float
    dist: MatchDistribution
    matches_a: int
    matches_b: int
    surface_matches_a: int
    surface_matches_b: int


def blended_probability(
    ratings: RatingSystem, a: int, b: int, surface: str, best_of: int
) -> tuple[float, float, float, float, float]:
    """(p_final, p_elo, p_saque_resto, pa, pb) sin calcular distribuciones completas."""
    p_elo = ratings.elo_probability(a, b, surface)
    if best_of == 5:
        # El Elo se estima sobre todos los formatos; al mejor de 5 el favorito gana más a menudo.
        p_elo = inv_logit(logit(p_elo) * 1.15)
    pa, pb = ratings.serve_probabilities(a, b, surface)
    p_sr = match_win_probability(pa, pb, best_of)
    w = ratings.params.elo_weight
    p_final = inv_logit(w * logit(p_elo) + (1 - w) * logit(p_sr))
    return p_final, p_elo, p_sr, pa, pb


def _mixture_win(pa: float, pb: float, best_of: int, sd: float) -> float:
    if sd <= 0:
        return match_win_probability(pa, pb, best_of)
    return sum(w * match_win_probability(pa + sd * x, pb - sd * x, best_of) for x, w in _NODES)


def _align_serves(pa: float, pb: float, target: float, best_of: int, sd: float) -> tuple[float, float]:
    """Desplaza ambas probabilidades de saque con un mismo delta hasta que el modelo
    de puntos (con forma del día) reproduzca la probabilidad de victoria objetivo."""
    lo, hi = -0.25, 0.25
    for _ in range(30):
        mid = (lo + hi) / 2
        if _mixture_win(pa + mid, pb - mid, best_of, sd) < target:
            lo = mid
        else:
            hi = mid
    delta = (lo + hi) / 2
    return pa + delta, pb - delta


def _mix(dists: list[tuple[float, MatchDistribution]]) -> MatchDistribution:
    def mix_dict(attr: str) -> dict:
        out: dict = defaultdict(float)
        for w, d in dists:
            for k, v in getattr(d, attr).items():
                out[k] += w * v
        return dict(out)

    def mix_num(attr: str) -> float:
        return sum(w * getattr(d, attr) for w, d in dists)

    return MatchDistribution(
        p_win_a=mix_num("p_win_a"),
        set_scores=mix_dict("set_scores"),
        total_sets=mix_dict("total_sets"),
        total_games=mix_dict("total_games"),
        games_a=mix_dict("games_a"),
        games_b=mix_dict("games_b"),
        total_breaks=mix_dict("total_breaks"),
        breaks_a=mix_dict("breaks_a"),
        breaks_b=mix_dict("breaks_b"),
        p_tiebreak=mix_num("p_tiebreak"),
        first_set_games=mix_dict("first_set_games"),
        p_first_set_tiebreak=mix_num("p_first_set_tiebreak"),
        p_a_first_set=mix_num("p_a_first_set"),
        best_of=dists[0][1].best_of,
    )


def distribution_with_form(sa: float, sb: float, best_of: int, sd: float) -> MatchDistribution:
    if sd <= 0:
        return match_distribution(sa, sb, best_of)
    return _mix([(w, match_distribution(sa + sd * x, sb - sd * x, best_of)) for x, w in _NODES])


def predict_match(
    ratings: RatingSystem,
    a: int,
    b: int,
    surface: str,
    best_of: int = 3,
    name_a: str | None = None,
    name_b: str | None = None,
) -> MatchPrediction:
    sd = ratings.params.form_sd
    p_final, p_elo, p_sr, pa, pb = blended_probability(ratings, a, b, surface, best_of)
    sa, sb = _align_serves(pa, pb, p_final, best_of, sd)
    dist = distribution_with_form(sa, sb, best_of, sd)
    st_a, st_b = ratings.players[a], ratings.players[b]
    return MatchPrediction(
        player_a=a,
        player_b=b,
        name_a=name_a or st_a.name or str(a),
        name_b=name_b or st_b.name or str(b),
        surface=surface,
        best_of=best_of,
        p_elo=p_elo,
        p_serve_return=p_sr,
        p_win_a=dist.p_win_a,
        serve_a=sa,
        serve_b=sb,
        dist=dist,
        matches_a=st_a.matches,
        matches_b=st_b.matches,
        surface_matches_a=st_a.surface_matches.get(surface, 0),
        surface_matches_b=st_b.surface_matches.get(surface, 0),
    )
