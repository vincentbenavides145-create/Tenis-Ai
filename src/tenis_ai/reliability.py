"""Fiabilidad histórica de los mercados más sólidos (para "Mejores predicciones").

Seis familias de mercados son las que más aciertan en el backtest fuera de muestra:
quiebres totales, quiebres de cada jugador, tie-break en el 1er set, juegos de cada
jugador, juegos del 1er set y juegos totales.

Para cada familia se guarda la diferencia entre el acierto real y la probabilidad
media del modelo (``gap``) en lo que ve el usuario. La puntuación de fiabilidad de
una selección es ``probabilidad + gap``: sirve solo para ORDENAR las mejores
predicciones; la probabilidad mostrada sigue siendo la del modelo.

``python backtest.py --save-reliability`` recalcula y guarda ``market_reliability.json``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

DEFAULT_RELIABILITY_PATH = Path(__file__).resolve().parents[2] / "market_reliability.json"

# grupo de mercado (markets.py) -> familia fiable
FAMILY_OF_GROUP = {
    "total_breaks": "total_breaks",
    "breaks_a": "player_breaks",
    "breaks_b": "player_breaks",
    "first_set_tiebreak": "first_set_tiebreak",
    "games_a": "player_games",
    "games_b": "player_games",
    "first_set_games": "first_set_games",
    "total_games": "total_games",
}
FAMILIES = tuple(dict.fromkeys(FAMILY_OF_GROUP.values()))


def family(group: str) -> str | None:
    return FAMILY_OF_GROUP.get(group)


def reliability_path() -> Path:
    env = os.getenv("TENIS_AI_RELIABILITY")
    return Path(env) if env else DEFAULT_RELIABILITY_PATH


def load_reliability(path: Path | None = None) -> dict[str, float]:
    """Diferencia (acierto real − prob. media) por familia. Sin archivo: 0 para todas."""
    gaps = {f: 0.0 for f in FAMILIES}
    target = path or reliability_path()
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        for fam, info in data.get("families", {}).items():
            if fam in gaps:
                gaps[fam] = float(info["gap"])
    except (OSError, ValueError, TypeError, KeyError):
        pass
    return gaps


def compute_reliability(by_group: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Agrega el resumen por grupo del backtest (lo mostrado) en familias, ponderando por muestra."""
    acc: dict[str, list[float]] = {}
    for group, block in by_group.items():
        fam = family(group)
        if fam is None or not block.get("count"):
            continue
        n = block["count"]
        tot = acc.setdefault(fam, [0, 0.0, 0.0])
        tot[0] += n
        tot[1] += block["mean_predicted"] * n
        tot[2] += block["hit_rate"] * n
    out: dict[str, dict[str, Any]] = {}
    for fam in FAMILIES:
        if fam not in acc:
            continue
        n, sp, sh = acc[fam]
        out[fam] = {
            "count": n,
            "mean_predicted": round(sp / n, 4),
            "hit_rate": round(sh / n, 4),
            "gap": round((sh - sp) / n, 4),
        }
    return out


def save_reliability(families: dict[str, dict[str, Any]], period: str, path: Path | None = None) -> Path:
    target = path or reliability_path()
    target.write_text(
        json.dumps({"period": period, "families": families}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return target


def reliability_score(item: dict[str, Any], gaps: dict[str, float]) -> float | None:
    fam = family(item.get("group", ""))
    if fam is None:
        return None
    return item["probability"] + gaps.get(fam, 0.0)
