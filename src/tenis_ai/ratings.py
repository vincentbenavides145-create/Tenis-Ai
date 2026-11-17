"""Estado de habilidad de los jugadores, actualizado partido a partido (sin mirar el futuro).

* Elo global y por superficie (K decreciente con la experiencia, estilo FiveThirtyEight).
* Saque/resto: puntos ganados al saque y al resto con ponderación temporal exponencial,
  peso reducido para otras superficies y contracción hacia la media del circuito.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date

from .data.sackmann import SURFACES, Match
from .params import TennisParams

TOUR_SERVE_DEFAULT = {"Hard": 0.640, "Clay": 0.615, "Grass": 0.665}


def _to_date(yyyymmdd: str) -> date:
    return date(int(yyyymmdd[:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:8]))


def _surface(surface: str) -> str:
    return surface if surface in SURFACES else "Hard"


@dataclass
class _Decayed:
    """Suma con decaimiento exponencial perezoso."""

    won: float = 0.0
    total: float = 0.0
    day: int = 0

    def decay_to(self, day: int, half_life: float) -> None:
        if day > self.day:
            factor = 0.5 ** ((day - self.day) / half_life)
            self.won *= factor
            self.total *= factor
            self.day = day

    def add(self, won: float, total: float, day: int, half_life: float) -> None:
        self.decay_to(day, half_life)
        self.won += won
        self.total += total


@dataclass
class PlayerState:
    name: str = ""
    matches: int = 0
    surface_matches: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    last_date: str = ""
    serve: dict[str, _Decayed] = field(default_factory=lambda: defaultdict(_Decayed))
    ret: dict[str, _Decayed] = field(default_factory=lambda: defaultdict(_Decayed))


class RatingSystem:
    def __init__(self, params: TennisParams) -> None:
        self.params = params
        self.elo: dict[tuple[int, str], float] = defaultdict(lambda: 1500.0)
        self.elo_n: dict[tuple[int, str], int] = defaultdict(int)
        self.players: dict[int, PlayerState] = defaultdict(PlayerState)
        self.tour_serve: dict[str, _Decayed] = defaultdict(_Decayed)
        self.last_day = 0

    # ------------------------------------------------------------------ Elo
    def _k(self, n: int) -> float:
        return self.params.elo_k_scale * 250.0 / ((n + 5) ** 0.4)

    def elo_rating(self, player: int, surface: str) -> float:
        w = self.params.elo_surface_weight
        return (1 - w) * self.elo[(player, "all")] + w * self.elo[(player, _surface(surface))]

    def elo_probability(self, a: int, b: int, surface: str) -> float:
        diff = self.elo_rating(a, surface) - self.elo_rating(b, surface)
        return 1.0 / (1.0 + 10 ** (-diff / 400.0))

    def _update_elo(self, winner: int, loser: int, surface: str) -> None:
        for key in ("all", _surface(surface)):
            rw, rl = self.elo[(winner, key)], self.elo[(loser, key)]
            expected = 1.0 / (1.0 + 10 ** ((rl - rw) / 400.0))
            self.elo[(winner, key)] = rw + self._k(self.elo_n[(winner, key)]) * (1 - expected)
            self.elo[(loser, key)] = rl - self._k(self.elo_n[(loser, key)]) * (1 - expected)
            self.elo_n[(winner, key)] += 1
            self.elo_n[(loser, key)] += 1

    # ---------------------------------------------------------- Saque/resto
    def tour_serve_avg(self, surface: str) -> float:
        s = _surface(surface)
        acc = self.tour_serve[s]
        prior = TOUR_SERVE_DEFAULT[s]
        return (acc.won + prior * 20000) / (acc.total + 20000)

    def _rate(self, sums: dict[str, _Decayed], surface: str, day: int, prior: float) -> tuple[float, float]:
        s = _surface(surface)
        hl, cw = self.params.half_life_days, self.params.cross_surface_weight
        won = total = 0.0
        for surf, acc in sums.items():
            acc.decay_to(day, hl)
            weight = 1.0 if surf == s else cw
            won += weight * acc.won
            total += weight * acc.total
        k = self.params.prior_points
        return (won + k * prior) / (total + k), total

    def serve_return(self, player: int, surface: str, day: int | None = None) -> tuple[float, float, float]:
        """(saque, resto, puntos efectivos de muestra) del jugador en esa superficie."""
        day = self.last_day if day is None else day
        avg = self.tour_serve_avg(surface)
        state = self.players[player]
        serve, n1 = self._rate(state.serve, surface, day, avg)
        ret, n2 = self._rate(state.ret, surface, day, 1 - avg)
        return serve, ret, min(n1, n2)

    def serve_probabilities(self, a: int, b: int, surface: str, day: int | None = None) -> tuple[float, float]:
        """Probabilidad de ganar un punto al saque de A y de B, ajustadas por el rival."""
        avg = self.tour_serve_avg(surface)
        sa, ra, _ = self.serve_return(a, surface, day)
        sb, rb, _ = self.serve_return(b, surface, day)
        pa = avg + (sa - avg) - (rb - (1 - avg))
        pb = avg + (sb - avg) - (ra - (1 - avg))
        return min(0.9, max(0.35, pa)), min(0.9, max(0.35, pb))

    # --------------------------------------------------------------- Update
    def update(self, match: Match) -> None:
        day = _to_date(match.date).toordinal()
        self.last_day = max(self.last_day, day)
        surface = _surface(match.surface)
        for pid, name in ((match.winner_id, match.winner_name), (match.loser_id, match.loser_name)):
            st = self.players[pid]
            st.name = name
            st.matches += 1
            st.surface_matches[surface] += 1
            st.last_date = match.date
        if match.completed or match.sets:
            self._update_elo(match.winner_id, match.loser_id, surface)
        hl = self.params.half_life_days
        w_sv, l_sv = match.serve_points(True), match.serve_points(False)
        if w_sv and l_sv:
            self.players[match.winner_id].serve[surface].add(w_sv[0], w_sv[1], day, hl)
            self.players[match.loser_id].serve[surface].add(l_sv[0], l_sv[1], day, hl)
            self.players[match.winner_id].ret[surface].add(l_sv[1] - l_sv[0], l_sv[1], day, hl)
            self.players[match.loser_id].ret[surface].add(w_sv[1] - w_sv[0], w_sv[1], day, hl)
            self.tour_serve[surface].add(w_sv[0] + l_sv[0], w_sv[1] + l_sv[1], day, 730.0)


def logit(p: float) -> float:
    p = min(1 - 1e-9, max(1e-9, p))
    return math.log(p / (1 - p))


def inv_logit(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))
