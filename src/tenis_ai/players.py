"""Resolución de nombres de jugador ("Alcaraz", "C. Alcaraz", "Carlos Alcaraz") a ids de Sackmann."""

from __future__ import annotations

import re
import unicodedata

from .ratings import RatingSystem


class PlayerNotFound(ValueError):
    pass


def normalize(name: str) -> str:
    text = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode().lower()
    text = re.sub(r"['`´]", "", text)  # O'Connell -> oconnell (como en el histórico)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]+", " ", text.replace("-", " "))).strip()


def resolve_player(ratings: RatingSystem, query: str) -> int:
    """Devuelve el id del jugador. Ante empates, el más activo recientemente."""
    q = normalize(query)
    if not q:
        raise PlayerNotFound("Nombre vacío.")
    tokens = q.split()
    # Abreviaturas con punto ("C.", "Dar.", "J. P.") son iniciales/prefijos del nombre, no apellidos.
    abbreviations = {normalize(t) for t in query.split() if t.endswith(".")}
    exact, by_last, by_initial = [], [], []
    for pid, st in ratings.players.items():
        name = normalize(st.name)
        if not name:
            continue
        parts = name.split()
        if name == q:
            exact.append(pid)
        elif len(tokens) == 1 and tokens[0] in parts[1:] + parts[-1:]:
            by_last.append(pid)
        elif len(tokens) >= 2:
            # "c alcaraz" / "alcaraz c" / "carlos alcaraz garfia"
            surname = [t for t in tokens if len(t) > 1 and t not in abbreviations]
            initials = [t for t in tokens if len(t) == 1 or t in abbreviations]
            if surname and all(s in parts for s in surname):
                if not initials or any(p.startswith(initials[0]) for p in parts if p not in surname):
                    by_initial.append(pid)
    for group in (exact, by_initial, by_last):
        if group:
            return max(group, key=lambda pid: (ratings.players[pid].last_date, ratings.players[pid].matches))
    raise PlayerNotFound(f"No encontré a '{query}' en el historial ATP.")
