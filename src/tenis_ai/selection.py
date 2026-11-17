"""Reglas de presentación (mismas que Deport-AI) y selección por valor esperado."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .reliability import load_reliability, reliability_score

MINIMUM_PROBABILITY = 0.60
MAXIMUM_PROBABILITY = 0.88
MIN_EDGE = 0.03  # EV mínimo (+3 %) cuando hay cuota
MAX_PER_CATEGORY = 4
MAX_BEST = 4  # tamaño de "Mejores predicciones"


def expected_value(probability: float, price: float) -> float:
    return probability * price - 1.0


def select_markets(
    candidates: list[dict[str, Any]],
    odds: dict[str, float] | None = None,
    minimum: float = MINIMUM_PROBABILITY,
    maximum: float = MAXIMUM_PROBABILITY,
    min_edge: float = MIN_EDGE,
    max_per_category: int = MAX_PER_CATEGORY,
) -> list[dict[str, Any]]:
    """1) rango 60–88 %; 2) con cuota, descarta EV < min_edge; 3) una selección por grupo
    (mayor EV si hay cuotas, si no la más probable); 4) máximo 4 por categoría, de mayor a menor."""
    eligible: list[dict[str, Any]] = []
    for cand in candidates:
        prob = cand["probability"]
        if prob < minimum or prob > maximum:
            continue
        item = dict(cand)
        price = (odds or {}).get(cand["id"])
        if price and price > 1.0:
            edge = round(expected_value(prob, price), 4)
            if edge < min_edge:
                continue
            item["odds"] = price
            item["edge"] = edge
        eligible.append(item)

    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in eligible:
        by_group[item["group"]].append(item)
    champions = []
    for items in by_group.values():
        priced = [i for i in items if i.get("edge") is not None]
        if priced:
            champions.append(max(priced, key=lambda i: (i["edge"], i["probability"])))
        else:
            champions.append(max(items, key=lambda i: i["probability"]))

    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in champions:
        by_cat[item["category"]].append(item)
    selected: list[dict[str, Any]] = []
    for items in by_cat.values():
        items.sort(key=lambda i: -i["probability"])
        selected.extend(items[:max_per_category])
    selected.sort(key=lambda i: -i["probability"])
    return selected


def best_predictions(
    selected: list[dict[str, Any]],
    gaps: dict[str, float] | None = None,
    limit: int = MAX_BEST,
) -> list[dict[str, Any]]:
    """Las selecciones (ya filtradas por las reglas) de los mercados más fiables del
    backtest, ordenadas por probabilidad corregida con su acierto histórico."""
    gaps = load_reliability() if gaps is None else gaps
    scored = [(reliability_score(i, gaps), i) for i in selected]
    scored = [(s, i) for s, i in scored if s is not None]
    scored.sort(key=lambda si: (-si[0], -si[1]["probability"]))
    return [i for _, i in scored[:limit]]


def top_pick(
    selected: list[dict[str, Any]],
    best: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Con cuotas: la de mayor EV. Sin cuotas: la primera de las mejores predicciones
    (mercado fiable); si no hay ninguna, la más probable."""
    if not selected:
        return None
    priced = [i for i in selected if i.get("edge") is not None]
    if priced:
        return max(priced, key=lambda i: (i["edge"], i["probability"]))
    if best:
        return best[0]
    return max(selected, key=lambda i: i["probability"])
