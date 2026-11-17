"""Construye el estado de ratings con todo el historial y genera reportes."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from .data.sackmann import Match, latest_surface_by_tournament, load_matches, normalize_tournament
from .formatter import format_report
from .markets import build_candidates
from .model import MatchPrediction, predict_match
from .params import TennisParams, load_params
from .players import resolve_player
from .ratings import RatingSystem
from .selection import select_markets, top_pick


# Torneos cuyo nombre en api-tennis.com no coincide con el del historial.
SURFACE_ALIASES = {
    "finals turin": "Hard",
    "next gen": "Hard",
    "united": "Hard",
    "davis": "Hard",
    "laver": "Hard",
}


@dataclass
class Report:
    prediction: MatchPrediction
    selected: list[dict[str, Any]]
    pick: dict[str, Any] | None
    markdown: str


class TennisEngine:
    def __init__(self, params: TennisParams | None = None, first_year: int = 2010) -> None:
        self.params = params or load_params()
        self.matches: list[Match] = load_matches(range(first_year, datetime.now().year + 1))
        self.ratings = RatingSystem(self.params)
        for match in self.matches:
            self.ratings.update(match)
        self.surfaces = latest_surface_by_tournament(self.matches)
        last = self.matches[-1].date if self.matches else "19700101"
        self.data_end = date(int(last[:4]), int(last[4:6]), int(last[6:8]))
        # Fin del histórico con estadísticas de saque (no cambia al actualizar el Elo por la API).
        self.history_end = self.data_end

    @property
    def data_age_days(self) -> int:
        return (date.today() - self.data_end).days

    def update_from_api(self, client: Any, until: date | None = None, chunk_days: int = 14) -> tuple[int, list[str]]:
        """Añade los resultados ATP posteriores al fin del histórico (solo Elo: la API no
        trae estadísticas de saque). Devuelve (partidos añadidos, incidencias)."""
        from datetime import timedelta

        from .data.sackmann import SetScore
        from .players import PlayerNotFound

        until = until or (date.today() - timedelta(days=1))
        start = self.data_end + timedelta(days=1)
        added, issues = 0, []
        while start <= until:
            stop = min(until, start + timedelta(days=chunk_days - 1))
            for row in client.finished_atp_results(start.isoformat(), stop.isoformat()):
                try:
                    winner = resolve_player(self.ratings, row["winner"])
                    loser = resolve_player(self.ratings, row["loser"])
                except PlayerNotFound:
                    issues.append(f"Sin historial: {row['winner']} vs {row['loser']} ({row['date']})")
                    continue
                surface = self.guess_surface(row["tournament"]) or "Hard"
                sets = tuple(SetScore(w, l) for w, l in row["sets"])
                self.ratings.update(
                    Match(
                        tourney_id=f"api-{row['date']}-{row['tournament']}",
                        tourney_name=row["tournament"],
                        surface=surface,
                        level="A",
                        date=row["date"].replace("-", ""),
                        round="",
                        match_num=added,
                        best_of=3,
                        winner_id=winner,
                        loser_id=loser,
                        winner_name=self.ratings.players[winner].name,
                        loser_name=self.ratings.players[loser].name,
                        score="",
                        sets=sets,
                        completed=bool(sets),
                    )
                )
                added += 1
            start = stop + timedelta(days=1)
        if added:
            self.data_end = until
        return added, issues

    def guess_surface(self, tournament: str) -> str | None:
        key = normalize_tournament(tournament)
        for alias, surface in SURFACE_ALIASES.items():
            if alias in key:
                return surface
        if key in self.surfaces:
            return self.surfaces[key]
        for name, surface in self.surfaces.items():
            if name and (name in key or key in name):
                return surface
        return None

    def report(
        self,
        player_a: str,
        player_b: str,
        surface: str,
        best_of: int = 3,
        odds: dict[str, float] | None = None,
        header_note: str = "",
    ) -> Report:
        a, b = resolve_player(self.ratings, player_a), resolve_player(self.ratings, player_b)
        pred = predict_match(self.ratings, a, b, surface, best_of)
        candidates = build_candidates(pred)
        selected = select_markets(candidates, odds=odds)
        pick = top_pick(selected)
        markdown = format_report(pred, selected, pick, self.data_age_days, header_note)
        return Report(pred, selected, pick, markdown)


def parse_matchup(text: str) -> tuple[str, str]:
    parts = re.split(r"\s+(?:vs\.?|v\.?|contra|-)\s+", text.strip(), maxsplit=1, flags=re.IGNORECASE)
    if len(parts) != 2:
        raise ValueError("Escribe el partido como 'Jugador A vs Jugador B'.")
    return parts[0].strip(), parts[1].strip()
