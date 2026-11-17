"""Modelo jerárquico de Markov exacto: punto → juego → tie-break → set → partido.

Entradas: `pa` = probabilidad de que A gane un punto con su saque, `pb` = la de B
con el suyo. Todo se calcula de forma exacta (sin simulación) y con caché.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache

_ROUND = 4  # precisión de la caché (las probabilidades de saque se redondean)


def _r(x: float) -> float:
    return round(min(0.98, max(0.20, x)), _ROUND)


@lru_cache(maxsize=None)
def hold_probability(p: float) -> float:
    """Probabilidad de ganar un juego al saque con probabilidad de punto `p` (forma cerrada)."""
    q = 1.0 - p
    deuce = p * p / (1.0 - 2.0 * p * q)
    return p**4 * (1 + 4 * q + 10 * q * q) + 20 * p**3 * q**3 * deuce


@lru_cache(maxsize=None)
def tiebreak_win(pa: float, pb: float, target: int, a_first: bool) -> float:
    """Probabilidad de que A gane un tie-break a `target` puntos (diferencia de 2)."""
    deuce = pa * (1 - pb) / (pa * (1 - pb) + (1 - pa) * pb)
    memo: dict[tuple[int, int], float] = {}

    def f(i: int, j: int) -> float:
        if (i, j) in memo:
            return memo[(i, j)]
        if i >= target and i - j >= 2:
            value = 1.0
        elif j >= target and j - i >= 2:
            value = 0.0
        elif i == j and i >= target - 1:
            value = deuce
        else:
            k = i + j
            a_serves = (((k + 1) // 2) % 2 == 0) == a_first
            p = pa if a_serves else 1.0 - pb
            value = p * f(i + 1, j) + (1 - p) * f(i, j + 1)
        memo[(i, j)] = value
        return value

    return f(0, 0)


def _set_over(ga: int, gb: int) -> bool:
    return (max(ga, gb) >= 6 and abs(ga - gb) >= 2) or max(ga, gb) == 7


@lru_cache(maxsize=None)
def set_outcomes(pa: float, pb: float, a_first: bool, tb_target: int = 7) -> tuple:
    """Distribución exacta del set: tuplas (ga, gb, ba, bb, prob).

    ba/bb = quiebres conseguidos por A/B. A 6-6 se juega tie-break (cuenta como juego: 7-6).
    """
    ha, hb = hold_probability(pa), hold_probability(pb)
    states: dict[tuple[int, int, int, int], float] = {(0, 0, 0, 0): 1.0}
    final: dict[tuple[int, int, int, int], float] = defaultdict(float)
    while states:
        nxt: dict[tuple[int, int, int, int], float] = defaultdict(float)
        for (ga, gb, ba, bb), p in states.items():
            a_serving = ((ga + gb) % 2 == 0) == a_first
            if ga == 6 and gb == 6:
                tw = tiebreak_win(pa, pb, tb_target, a_serving)
                final[(7, 6, ba, bb)] += p * tw
                final[(6, 7, ba, bb)] += p * (1 - tw)
                continue
            if a_serving:
                moves = (((ga + 1, gb, ba, bb), ha), ((ga, gb + 1, ba, bb + 1), 1 - ha))
            else:
                moves = (((ga + 1, gb, ba + 1, bb), 1 - hb), ((ga, gb + 1, ba, bb), hb))
            for key, q in moves:
                bucket = final if _set_over(key[0], key[1]) else nxt
                bucket[key] += p * q
        states = nxt
    return tuple((ga, gb, ba, bb, p) for (ga, gb, ba, bb), p in final.items())


@lru_cache(maxsize=200_000)
def _set_summary(pa: float, pb: float, a_first: bool, tb_target: int) -> tuple:
    """((gana A, A saca primero el siguiente set), prob). DP ligera solo sobre (ga, gb)."""
    ha, hb = hold_probability(pa), hold_probability(pb)
    states: dict[tuple[int, int], float] = {(0, 0): 1.0}
    out: dict[tuple[bool, bool], float] = defaultdict(float)
    while states:
        nxt: dict[tuple[int, int], float] = defaultdict(float)
        for (ga, gb), p in states.items():
            a_serving = ((ga + gb) % 2 == 0) == a_first
            if ga == 6 and gb == 6:
                tw = tiebreak_win(pa, pb, tb_target, a_serving)
                next_a_first = not a_serving
                out[(True, next_a_first)] += p * tw
                out[(False, next_a_first)] += p * (1 - tw)
                continue
            win = ha if a_serving else 1 - hb
            for (na, nb), q in (((ga + 1, gb), win), ((ga, gb + 1), 1 - win)):
                if _set_over(na, nb):
                    out[(na > nb, ((na + nb) % 2 == 0) == a_first)] += p * q
                else:
                    nxt[(na, nb)] += p * q
        states = nxt
    return tuple(out.items())


def _tb_target(sets_played: int, best_of: int) -> int:
    # Grand Slams (al mejor de 5): set decisivo con tie-break a 10 a 6-6.
    return 10 if (best_of == 5 and sets_played == best_of - 1) else 7


def match_win_probability(pa: float, pb: float, best_of: int = 3) -> float:
    """Probabilidad de que A gane el partido (sorteo de saque 50/50). Vía rápida."""
    return _match_win(_r(pa), _r(pb), best_of)


@lru_cache(maxsize=500_000)
def _match_win(pa: float, pb: float, best_of: int) -> float:
    need = best_of // 2 + 1
    total = 0.0
    for a_first0 in (True, False):
        states: dict[tuple[int, int, bool], float] = {(0, 0, a_first0): 0.5}
        while states:
            nxt: dict[tuple[int, int, bool], float] = defaultdict(float)
            for (sa, sb, af), p in states.items():
                for (a_won, naf), q in _set_summary(pa, pb, af, _tb_target(sa + sb, best_of)):
                    na, nb = sa + a_won, sb + (not a_won)
                    if na == need:
                        total += p * q
                    elif nb < need:
                        nxt[(na, nb, naf)] += p * q
            states = nxt
    return total


@dataclass
class MatchDistribution:
    p_win_a: float
    set_scores: dict[tuple[int, int], float]
    total_sets: dict[int, float]
    total_games: dict[int, float]
    games_a: dict[int, float]
    games_b: dict[int, float]
    total_breaks: dict[int, float]
    breaks_a: dict[int, float]
    breaks_b: dict[int, float]
    p_tiebreak: float
    first_set_games: dict[int, float]
    p_first_set_tiebreak: float
    p_a_first_set: float
    best_of: int = 3
    extra: dict[str, float] = field(default_factory=dict)

    @staticmethod
    def over(dist: dict[int, float], line: float) -> float:
        return sum(p for k, p in dist.items() if k > line)

    @staticmethod
    def expected(dist: dict[int, float]) -> float:
        return sum(k * p for k, p in dist.items())


def _convolve(acc: dict, outcomes: dict) -> dict:
    out: dict = defaultdict(float)
    for (x1, y1), p in acc.items():
        for (x2, y2), q in outcomes.items():
            out[(x1 + x2, y1 + y2)] += p * q
    return out


def match_distribution(pa: float, pb: float, best_of: int = 3) -> MatchDistribution:
    """Distribuciones exactas del partido (promedia quién saca primero)."""
    pa, pb = _r(pa), _r(pb)
    need = best_of // 2 + 1
    set_scores: dict[tuple[int, int], float] = defaultdict(float)
    games: dict[tuple[int, int], float] = defaultdict(float)
    breaks: dict[tuple[int, int], float] = defaultdict(float)
    no_tb = 0.0
    first_set_games: dict[int, float] = defaultdict(float)
    p_first_tb = p_a_first_set = 0.0

    for a_first0 in (True, False):
        for ga, gb, _ba, _bb, p in set_outcomes(pa, pb, a_first0, 7):
            first_set_games[ga + gb] += 0.5 * p
            if ga + gb == 13:
                p_first_tb += 0.5 * p
            if ga > gb:
                p_a_first_set += 0.5 * p

        # Para cada estado (sets A, sets B, A saca primero) se guardan distribuciones
        # acumuladas (no normalizadas) de juegos y quiebres, y la masa sin tie-break.
        states: dict[tuple[int, int, bool], tuple[dict, dict, float]] = {
            (0, 0, a_first0): ({(0, 0): 0.5}, {(0, 0): 0.5}, 0.5)
        }
        while states:
            nxt: dict[tuple[int, int, bool], list] = {}
            for (sa, sb, af), (gdist, bdist, notb_mass) in states.items():
                by_g: dict = defaultdict(lambda: defaultdict(float))
                by_b: dict = defaultdict(lambda: defaultdict(float))
                notb_set: dict = defaultdict(float)
                for ga, gb, ba, bb, q in set_outcomes(pa, pb, af, _tb_target(sa + sb, best_of)):
                    k2 = (ga > gb, ((ga + gb) % 2 == 0) == af)
                    by_g[k2][(ga, gb)] += q
                    by_b[k2][(ba, bb)] += q
                    if ga + gb != 13:
                        notb_set[k2] += q
                mass = sum(gdist.values())
                for k2, g_out in by_g.items():
                    a_won, naf = k2
                    na, nb = sa + a_won, sb + (not a_won)
                    gacc = _convolve(gdist, g_out)
                    bacc = _convolve(bdist, by_b[k2])
                    nt = notb_mass * notb_set[k2]
                    if na == need or nb == need:
                        set_scores[(na, nb)] += mass * sum(g_out.values())
                        for k3, v in gacc.items():
                            games[k3] += v
                        for k3, v in bacc.items():
                            breaks[k3] += v
                        no_tb += nt
                    else:
                        key = (na, nb, naf)
                        if key not in nxt:
                            nxt[key] = [defaultdict(float), defaultdict(float), 0.0]
                        for k3, v in gacc.items():
                            nxt[key][0][k3] += v
                        for k3, v in bacc.items():
                            nxt[key][1][k3] += v
                        nxt[key][2] += nt
            states = {k: (v[0], v[1], v[2]) for k, v in nxt.items()}

    def marg(joint: dict, fn) -> dict[int, float]:
        out: dict[int, float] = defaultdict(float)
        for k, p in joint.items():
            out[fn(k)] += p
        return dict(out)

    total_sets: dict[int, float] = defaultdict(float)
    for (sa, sb), p in set_scores.items():
        total_sets[sa + sb] += p
    return MatchDistribution(
        p_win_a=sum(p for (sa, _sb), p in set_scores.items() if sa == need),
        set_scores=dict(set_scores),
        total_sets=dict(total_sets),
        total_games=marg(games, lambda k: k[0] + k[1]),
        games_a=marg(games, lambda k: k[0]),
        games_b=marg(games, lambda k: k[1]),
        total_breaks=marg(breaks, lambda k: k[0] + k[1]),
        breaks_a=marg(breaks, lambda k: k[0]),
        breaks_b=marg(breaks, lambda k: k[1]),
        p_tiebreak=1.0 - no_tb,
        first_set_games=dict(first_set_games),
        p_first_set_tiebreak=p_first_tb,
        p_a_first_set=p_a_first_set,
        best_of=best_of,
    )
