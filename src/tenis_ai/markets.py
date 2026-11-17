"""Catálogo de mercados ATP (sin hándicaps) y su liquidación contra el resultado real.

Cada candidato: id, category, group (subfamilia: una sola selección por grupo),
market, selection, probability, projection, explanation y, para liquidar, `settle`.
"""

from __future__ import annotations

from typing import Any

from .data.sackmann import Match
from .markov import MatchDistribution, hold_probability
from .model import MatchPrediction

CATEGORIES = ("Ganador", "Sets", "Juegos", "Tie-breaks", "Quiebres")


def _lines(best_of: int, bo3: list[float], bo5: list[float]) -> list[float]:
    return bo5 if best_of == 5 else bo3


def _over(dist: dict[int, float], line: float) -> float:
    return MatchDistribution.over(dist, line)


def build_candidates(pred: MatchPrediction) -> list[dict[str, Any]]:
    d, a, b, bo = pred.dist, pred.name_a, pred.name_b, pred.best_of
    need = bo // 2 + 1
    hold_a, hold_b = hold_probability(pred.serve_a), hold_probability(pred.serve_b)
    exp_games = d.expected(d.total_games)
    exp_ga, exp_gb = d.expected(d.games_a), d.expected(d.games_b)
    exp_ba, exp_bb = d.expected(d.breaks_a), d.expected(d.breaks_b)
    out: list[dict[str, Any]] = []

    def add(cid, category, group, market, selection, prob, explanation, settle, projection=None):
        out.append(
            {
                "id": cid,
                "category": category,
                "group": group,
                "market": market,
                "selection": selection,
                "probability": round(float(prob), 4),
                "projection": projection,
                "explanation": explanation,
                "settle": settle,
            }
        )

    # --- Ganador
    fav, fav_hold = (a, hold_a) if d.p_win_a >= 0.5 else (b, hold_b)
    add("winner_a", "Ganador", "winner", "Ganador del partido", f"{a} gana", d.p_win_a,
        f"Porque {a} retiene su saque el {hold_a * 100:.0f}% de los juegos frente al {hold_b * 100:.0f}% de {b}.",
        ("winner", "a"))
    add("winner_b", "Ganador", "winner", "Ganador del partido", f"{b} gana", 1 - d.p_win_a,
        f"Porque {b} retiene su saque el {hold_b * 100:.0f}% de los juegos frente al {hold_a * 100:.0f}% de {a}.",
        ("winner", "b"))

    # --- Sets
    for (sa, sb), p in sorted(d.set_scores.items()):
        winner = a if sa == need else b
        add(f"set_score_{sa}_{sb}", "Sets", "set_score", "Resultado exacto en sets", f"{winner} gana {max(sa, sb)}-{min(sa, sb)}", p,
            f"Porque el marcador {sa}-{sb} es el más consistente con un {fav} favorito que retiene el {fav_hold * 100:.0f}% de sus saques.",
            ("set_score", (sa, sb)))
    exp_sets = d.expected(d.total_sets)
    for line in _lines(bo, [2.5], [3.5, 4.5]):
        p_over = _over(d.total_sets, line)
        add(f"sets_over_{line}", "Sets", "total_sets", "Total de sets", f"Más de {line} sets", p_over,
            f"Porque el modelo proyecta {exp_sets:.2f} sets en promedio.", ("total_sets", ">", line), exp_sets)
        add(f"sets_under_{line}", "Sets", "total_sets", "Total de sets", f"Menos de {line} sets", 1 - p_over,
            f"Porque el modelo proyecta {exp_sets:.2f} sets en promedio.", ("total_sets", "<", line), exp_sets)
    p_a_set = sum(p for (sa, _), p in d.set_scores.items() if sa >= 1)
    p_b_set = sum(p for (_, sb), p in d.set_scores.items() if sb >= 1)
    add("a_wins_set", "Sets", "a_wins_set", "Gana al menos un set", f"{a} gana al menos un set", p_a_set,
        f"Porque {a} gana el {pred.serve_a * 100:.1f}% de los puntos con su saque, suficiente para competir un set.", ("wins_set", "a"))
    add("b_wins_set", "Sets", "b_wins_set", "Gana al menos un set", f"{b} gana al menos un set", p_b_set,
        f"Porque {b} gana el {pred.serve_b * 100:.1f}% de los puntos con su saque, suficiente para competir un set.", ("wins_set", "b"))
    add("first_set_a", "Sets", "first_set", "Ganador del 1er set", f"{a} gana el 1er set", d.p_a_first_set,
        f"Porque {a} parte con ventaja en el intercambio de saques del primer set.", ("first_set", "a"))
    add("first_set_b", "Sets", "first_set", "Ganador del 1er set", f"{b} gana el 1er set", 1 - d.p_a_first_set,
        f"Porque {b} parte con ventaja en el intercambio de saques del primer set.", ("first_set", "b"))

    # --- Juegos
    for line in _lines(bo, [18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5], [33.5, 35.5, 36.5, 37.5, 38.5, 39.5, 41.5]):
        p_over = _over(d.total_games, line)
        add(f"games_over_{line}", "Juegos", "total_games", "Juegos totales", f"Más de {line} juegos", p_over,
            f"Porque el modelo proyecta {exp_games:.1f} juegos con saques retenidos al {hold_a * 100:.0f}% y {hold_b * 100:.0f}%.",
            ("total_games", ">", line), exp_games)
        add(f"games_under_{line}", "Juegos", "total_games", "Juegos totales", f"Menos de {line} juegos", 1 - p_over,
            f"Porque el modelo proyecta {exp_games:.1f} juegos con saques retenidos al {hold_a * 100:.0f}% y {hold_b * 100:.0f}%.",
            ("total_games", "<", line), exp_games)
    for side, name, dist, exp in (("a", a, d.games_a, exp_ga), ("b", b, d.games_b, exp_gb)):
        for line in _lines(bo, [8.5, 9.5, 10.5, 11.5, 12.5, 13.5], [14.5, 16.5, 18.5, 20.5, 22.5]):
            p_over = _over(dist, line)
            add(f"games_{side}_over_{line}", "Juegos", f"games_{side}", f"Juegos de {name}", f"{name} Más de {line} juegos", p_over,
                f"Porque el modelo proyecta {exp:.1f} juegos ganados por {name}.", (f"games_{side}", ">", line), exp)
            add(f"games_{side}_under_{line}", "Juegos", f"games_{side}", f"Juegos de {name}", f"{name} Menos de {line} juegos", 1 - p_over,
                f"Porque el modelo proyecta {exp:.1f} juegos ganados por {name}.", (f"games_{side}", "<", line), exp)
    exp_fs = d.expected(d.first_set_games)
    for line in (8.5, 9.5, 10.5):
        p_over = _over(d.first_set_games, line)
        add(f"fs_games_over_{line}", "Juegos", "first_set_games", "Juegos del 1er set", f"1er set Más de {line} juegos", p_over,
            f"Porque el modelo proyecta {exp_fs:.1f} juegos en el primer set.", ("first_set_games", ">", line), exp_fs)
        add(f"fs_games_under_{line}", "Juegos", "first_set_games", "Juegos del 1er set", f"1er set Menos de {line} juegos", 1 - p_over,
            f"Porque el modelo proyecta {exp_fs:.1f} juegos en el primer set.", ("first_set_games", "<", line), exp_fs)

    # --- Tie-breaks
    add("tb_yes", "Tie-breaks", "tiebreak", "Tie-break en el partido", "Habrá tie-break en el partido", d.p_tiebreak,
        f"Porque ambos retienen su saque con frecuencia ({hold_a * 100:.0f}% y {hold_b * 100:.0f}%).", ("tiebreak", True))
    add("tb_no", "Tie-breaks", "tiebreak", "Tie-break en el partido", "No habrá tie-break en el partido", 1 - d.p_tiebreak,
        f"Porque la diferencia al saque ({hold_a * 100:.0f}% frente a {hold_b * 100:.0f}%) favorece quiebres que cierran los sets.",
        ("tiebreak", False))
    add("fs_tb_yes", "Tie-breaks", "first_set_tiebreak", "Tie-break en el 1er set", "Habrá tie-break en el 1er set", d.p_first_set_tiebreak,
        "Porque el primer set se proyecta muy parejo al saque.", ("first_set_tiebreak", True))
    add("fs_tb_no", "Tie-breaks", "first_set_tiebreak", "Tie-break en el 1er set", "No habrá tie-break en el 1er set", 1 - d.p_first_set_tiebreak,
        "Porque se espera al menos un quiebre decisivo en el primer set.", ("first_set_tiebreak", False))

    # --- Quiebres
    exp_tb = d.expected(d.total_breaks)
    for line in _lines(bo, [2.5, 3.5, 4.5, 5.5, 6.5], [4.5, 5.5, 6.5, 7.5, 8.5]):
        p_over = _over(d.total_breaks, line)
        add(f"breaks_over_{line}", "Quiebres", "total_breaks", "Quiebres totales", f"Más de {line} quiebres", p_over,
            f"Porque el modelo proyecta {exp_tb:.1f} quiebres de saque en el partido.", ("total_breaks", ">", line), exp_tb)
        add(f"breaks_under_{line}", "Quiebres", "total_breaks", "Quiebres totales", f"Menos de {line} quiebres", 1 - p_over,
            f"Porque el modelo proyecta {exp_tb:.1f} quiebres de saque en el partido.", ("total_breaks", "<", line), exp_tb)
    for side, name, dist, exp in (("a", a, d.breaks_a, exp_ba), ("b", b, d.breaks_b, exp_bb)):
        for line in (0.5, 1.5, 2.5, 3.5):
            p_over = _over(dist, line)
            label = f"{name} quiebra al menos una vez" if line == 0.5 else f"{name} Más de {line} quiebres"
            add(f"breaks_{side}_over_{line}", "Quiebres", f"breaks_{side}", f"Quiebres de {name}", label, p_over,
                f"Porque el modelo proyecta {exp:.1f} quiebres para {name}.", (f"breaks_{side}", ">", line), exp)
            if line >= 1.5:
                add(f"breaks_{side}_under_{line}", "Quiebres", f"breaks_{side}", f"Quiebres de {name}", f"{name} Menos de {line} quiebres",
                    1 - p_over, f"Porque el modelo proyecta {exp:.1f} quiebres para {name}.", (f"breaks_{side}", "<", line), exp)
    return out


def actual_outcomes(match: Match, a_is_winner: bool) -> dict[str, Any] | None:
    """Resultado real desde la perspectiva A/B. None si el partido no se completó."""
    if not match.completed:
        return None
    ws = [(s.winner_games, s.loser_games) for s in match.sets]
    sets = ws if a_is_winner else [(y, x) for x, y in ws]
    need = match.best_of // 2 + 1
    sets_a = sum(1 for x, y in sets if x > y)
    sets_b = len(sets) - sets_a
    if max(sets_a, sets_b) != need:
        return None
    outcome = {
        "winner": "a" if sets_a > sets_b else "b",
        "set_score": (sets_a, sets_b),
        "total_sets": len(sets),
        "games_a": sum(x for x, _ in sets),
        "games_b": sum(y for _, y in sets),
        "first_set": "a" if sets[0][0] > sets[0][1] else "b",
        "first_set_games": sum(sets[0]),
        "tiebreak": any({x, y} == {7, 6} for x, y in sets),
        "first_set_tiebreak": set(sets[0]) == {7, 6},
        "wins_set_a": sets_a >= 1,
        "wins_set_b": sets_b >= 1,
        # Sets con ventaja (p. ej. 13-11) no son compatibles con el modelo de juegos.
        "games_supported": all(max(x, y) <= 7 for x, y in sets),
    }
    outcome["total_games"] = outcome["games_a"] + outcome["games_b"]
    br = match.breaks()
    if br is not None:
        ba, bb = br if a_is_winner else (br[1], br[0])
        outcome.update({"breaks_a": ba, "breaks_b": bb, "total_breaks": ba + bb})
    return outcome


def settle(candidate: dict[str, Any], outcome: dict[str, Any]) -> bool | None:
    """True/False si la selección gana/pierde; None si no se puede liquidar."""
    rule = candidate["settle"]
    kind = rule[0]
    if kind == "winner":
        return outcome["winner"] == rule[1]
    if kind == "set_score":
        return outcome["set_score"] == rule[1]
    if kind == "wins_set":
        return outcome[f"wins_set_{rule[1]}"]
    if kind == "first_set":
        return outcome["first_set"] == rule[1]
    if kind in ("tiebreak", "first_set_tiebreak"):
        return outcome[kind] == rule[1]
    if kind in ("total_games", "games_a", "games_b", "first_set_games") and not outcome["games_supported"]:
        return None
    value = outcome.get(kind)
    if value is None:
        return None
    op, line = rule[1], rule[2]
    return value > line if op == ">" else value < line
