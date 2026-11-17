"""Reporte Markdown en el estilo de Deport-AI: sin tablas, una línea "Porque ..." por selección."""

from __future__ import annotations

from typing import Any

from .model import MatchPrediction

SURFACE_ES = {"Hard": "Dura", "Clay": "Tierra batida", "Grass": "Hierba"}
HEADERS = [
    ("Ganador", "#### 🏆 Ganador del partido"),
    ("Sets", "#### 🎾 Sets"),
    ("Juegos", "#### 🔢 Juegos"),
    ("Tie-breaks", "#### ⚡ Tie-breaks"),
    ("Quiebres", "#### 💥 Quiebres de saque"),
]


def confidence(pred: MatchPrediction, data_age_days: int | None = None) -> tuple[str, str]:
    low = min(pred.matches_a, pred.matches_b)
    low_surface = min(pred.surface_matches_a, pred.surface_matches_b)
    if low >= 50 and low_surface >= 15:
        level = "🟢 Alta"
    elif low >= 20:
        level = "🟡 Media"
    else:
        level = "🔴 Baja"
    detail = f"{pred.matches_a} y {pred.matches_b} partidos ATP en el historial, {pred.surface_matches_a} y {pred.surface_matches_b} en esta superficie"
    if data_age_days is not None and data_age_days > 45:
        level = "🔴 Baja" if level != "🟢 Alta" else "🟡 Media"
        detail += f"; los datos históricos terminan hace {data_age_days} días"
    return level, detail


def _line(item: dict[str, Any]) -> list[str]:
    suffix = ""
    if item.get("odds") and item.get("edge") is not None:
        suffix = f" · cuota {item['odds']:.2f} · EV {item['edge'] * 100:+.1f}%"
    return [f"• **{item['selection']} — {item['probability'] * 100:.1f}%**{suffix}  ", f"  *{item['explanation']}*  "]


def format_report(
    pred: MatchPrediction,
    selected: list[dict[str, Any]],
    pick: dict[str, Any] | None,
    data_age_days: int | None = None,
    header_note: str = "",
    best: list[dict[str, Any]] | None = None,
) -> str:
    d = pred.dist
    level, detail = confidence(pred, data_age_days)
    lines = [
        f"# 🎾 {pred.name_a} vs {pred.name_b}",
        f"**Superficie:** {SURFACE_ES.get(pred.surface, pred.surface)} · al mejor de {pred.best_of} sets"
        + (f" · {header_note}" if header_note else ""),
        f"**Confianza:** {level} ({detail})",
        "",
    ]
    if pick:
        lines += ["### ⭐ PICK RECOMENDADO POR EL MODELO (AI Top Pick)", *_line(pick), ""]
    if best:
        lines.append("### 🏅 MEJORES PREDICCIONES DEL PARTIDO")
        for item in best:
            lines += _line(item)
        lines.append("")
    for category, header in HEADERS:
        items = [i for i in selected if i["category"] == category]
        if not items:
            continue
        lines.append(header)
        for item in items:
            lines += _line(item)
        lines.append("")
    best_score = max(d.set_scores.items(), key=lambda kv: kv[1])
    (sa, sb), p_score = best_score
    lines += [
        "#### 📊 Proyecciones numéricas",
        f"• **Probabilidad de victoria:** {pred.name_a} {d.p_win_a * 100:.1f}% | {pred.name_b} {(1 - d.p_win_a) * 100:.1f}%  ",
        f"• **Juegos estimados:** {d.expected(d.total_games):.1f} ({pred.name_a}: {d.expected(d.games_a):.1f} | {pred.name_b}: {d.expected(d.games_b):.1f})  ",
        f"• **Quiebres estimados:** {d.expected(d.total_breaks):.1f} ({pred.name_a}: {d.expected(d.breaks_a):.1f} | {pred.name_b}: {d.expected(d.breaks_b):.1f})  ",
        f"• **Probabilidad de tie-break:** {d.p_tiebreak * 100:.1f}%  ",
        f"• **Marcador en sets más probable:** {sa}-{sb} ({p_score * 100:.1f}%)",
    ]
    return "\n".join(lines)
