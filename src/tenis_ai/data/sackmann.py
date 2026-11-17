"""Partidos ATP históricos en formato Jeff Sackmann (licencia CC BY-NC-SA 4.0).

El repositorio original de GitHub ya no está disponible; se usa el espejo de archivo
publicado en Hugging Face. Los CSV se guardan en caché en `data/raw/`.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import requests

MIRROR_URL = (
    "https://huggingface.co/datasets/Aneeshers/tennis-sackmann-archive/resolve/main/atp/atp_matches_{year}.csv"
)
DEFAULT_CACHE = Path(__file__).resolve().parents[3] / "data" / "raw"

ROUND_ORDER = {
    "Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3, "ER": 4, "RR": 5, "R128": 6, "R64": 7,
    "R32": 8, "R16": 9, "QF": 10, "SF": 11, "BR": 12, "F": 13,
}
SURFACES = ("Hard", "Clay", "Grass")
_SET_RE = re.compile(r"^(\d+)-(\d+)(?:\((\d+)\))?$")


@dataclass(frozen=True)
class SetScore:
    winner_games: int
    loser_games: int

    @property
    def tiebreak(self) -> bool:
        return {self.winner_games, self.loser_games} == {7, 6}


@dataclass(frozen=True)
class Match:
    tourney_id: str
    tourney_name: str
    surface: str
    level: str
    date: str  # AAAAMMDD (inicio del torneo)
    round: str
    match_num: int
    best_of: int
    winner_id: int
    loser_id: int
    winner_name: str
    loser_name: str
    score: str
    sets: tuple[SetScore, ...]
    completed: bool
    stats: dict[str, int] = field(default_factory=dict, hash=False, compare=False)

    @property
    def order_key(self) -> tuple:
        return (self.date, self.tourney_id, ROUND_ORDER.get(self.round, 5), self.match_num)

    def serve_points(self, winner: bool) -> tuple[int, int] | None:
        """(puntos ganados al saque, puntos de saque) del ganador o del perdedor."""
        prefix = "w_" if winner else "l_"
        try:
            won = self.stats[prefix + "1stWon"] + self.stats[prefix + "2ndWon"]
            total = self.stats[prefix + "svpt"]
        except KeyError:
            return None
        return (won, total) if total > 0 and 0 <= won <= total else None

    def breaks(self) -> tuple[int, int] | None:
        """(quiebres del ganador, quiebres del perdedor) a partir de bolas de quiebre."""
        try:
            by_winner = self.stats["l_bpFaced"] - self.stats["l_bpSaved"]
            by_loser = self.stats["w_bpFaced"] - self.stats["w_bpSaved"]
        except KeyError:
            return None
        if by_winner < 0 or by_loser < 0:
            return None
        return by_winner, by_loser


def parse_score(score: str) -> tuple[tuple[SetScore, ...], bool]:
    """Convierte '6-3 3-6 7-6(5)' en sets; completed=False si hay RET/W/O/DEF o formato raro."""
    sets: list[SetScore] = []
    completed = True
    for token in (score or "").replace("  ", " ").split():
        match = _SET_RE.match(token)
        if match:
            sets.append(SetScore(int(match.group(1)), int(match.group(2))))
        else:
            completed = False  # RET, W/O, DEF, [10-8], ABD, etc.
    if not sets:
        completed = False
    return tuple(sets), completed


_STAT_COLUMNS = (
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_SvGms", "w_bpSaved", "w_bpFaced",
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_SvGms", "l_bpSaved", "l_bpFaced",
)


def _row_to_match(row: dict[str, str]) -> Match | None:
    try:
        winner_id, loser_id = int(row["winner_id"]), int(row["loser_id"])
    except (KeyError, ValueError):
        return None
    sets, completed = parse_score(row.get("score", ""))
    stats: dict[str, int] = {}
    for col in _STAT_COLUMNS:
        value = row.get(col, "")
        if value not in ("", None):
            try:
                stats[col] = int(float(value))
            except ValueError:
                pass
    try:
        best_of = int(row.get("best_of") or 3)
    except ValueError:
        best_of = 3
    try:
        match_num = int(row.get("match_num") or 0)
    except ValueError:
        match_num = 0
    return Match(
        tourney_id=row.get("tourney_id", ""),
        tourney_name=row.get("tourney_name", ""),
        surface=row.get("surface") or "Hard",
        level=row.get("tourney_level", ""),
        date=row.get("tourney_date", ""),
        round=row.get("round", ""),
        match_num=match_num,
        best_of=best_of,
        winner_id=winner_id,
        loser_id=loser_id,
        winner_name=row.get("winner_name", ""),
        loser_name=row.get("loser_name", ""),
        score=row.get("score", ""),
        sets=sets,
        completed=completed,
        stats=stats,
    )


def download_year(year: int, cache_dir: Path = DEFAULT_CACHE, refresh: bool = False) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"atp_matches_{year}.csv"
    if path.exists() and not refresh:
        return path
    response = requests.get(MIRROR_URL.format(year=year), timeout=120)
    response.raise_for_status()
    path.write_bytes(response.content)
    return path


def load_matches(
    years: Iterable[int],
    cache_dir: Path = DEFAULT_CACHE,
    refresh_years: Iterable[int] = (),
) -> list[Match]:
    """Carga los partidos (descarga lo que falte) ordenados cronológicamente."""
    refresh = set(refresh_years)
    matches: list[Match] = []
    for year in years:
        try:
            path = download_year(year, cache_dir, refresh=year in refresh)
        except requests.RequestException:
            path = cache_dir / f"atp_matches_{year}.csv"
            if not path.exists():
                continue
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                match = _row_to_match(row)
                if match is not None:
                    matches.append(match)
    matches.sort(key=lambda m: m.order_key)
    return matches


def latest_surface_by_tournament(matches: Iterable[Match]) -> dict[str, str]:
    """Nombre de torneo normalizado -> superficie más reciente (para el calendario en vivo)."""
    out: dict[str, str] = {}
    for match in matches:
        if match.surface in SURFACES:
            out[normalize_tournament(match.tourney_name)] = match.surface
    return out


def normalize_tournament(name: str) -> str:
    import unicodedata

    text = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode().lower()
    text = re.sub(r"\b(atp|masters|open|championships?|cup|presented|by|the|\d{4})\b", " ", text)
    return re.sub(r"[^a-z]+", " ", text).strip()
