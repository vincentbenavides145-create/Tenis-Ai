"""Histórico permanente de predicciones y su comprobación con los resultados reales.

Cada partido predicho es una línea JSON en `history/predictions.jsonl` (dentro del
repositorio, para que se conserve: las sesiones en la nube son temporales). Si se
vuelve a predecir el mismo partido antes de que empiece, se guarda la última versión.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from .markets import settle
from .players import normalize

DEFAULT_HISTORY = Path(__file__).resolve().parents[2] / "history" / "predictions.jsonl"
SHOWN_BINS = [(0.60, 0.65), (0.65, 0.70), (0.70, 0.75), (0.75, 0.80), (0.80, 0.885)]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _jsonable_rule(rule: Any) -> Any:
    return [list(x) if isinstance(x, tuple) else x for x in rule]


def _rule_from_json(rule: list) -> tuple:
    # set_score se guarda como lista [2, 1]; settle() compara con una tupla.
    return tuple(tuple(x) if isinstance(x, list) else x for x in rule)


class HistoryStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else DEFAULT_HISTORY

    # ------------------------------------------------------------ lectura/escritura
    def load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records

    def write(self, records: Iterable[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        ordered = sorted(records, key=lambda r: (r["match_date"], r["key"]))
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ordered), encoding="utf-8")
        tmp.replace(self.path)

    # --------------------------------------------------------------------- guardar
    def save_prediction(
        self,
        report: Any,
        match_date: str,
        tournament: str = "",
        event_key: str | None = None,
        source: str = "manual",
    ) -> dict[str, Any]:
        pred = report.prediction
        key = event_key or f"{match_date}_{pred.player_a}_{pred.player_b}"
        pick_id = report.pick["id"] if report.pick else None
        best_ids = {s["id"] for s in getattr(report, "best", None) or []}
        record = {
            "key": key,
            "event_key": event_key,
            "source": source,
            "created_at": _now(),
            "match_date": match_date,
            "tournament": tournament,
            "surface": pred.surface,
            "best_of": pred.best_of,
            "player_a": {"id": pred.player_a, "name": pred.name_a},
            "player_b": {"id": pred.player_b, "name": pred.name_b},
            "p_win_a": round(pred.p_win_a, 4),
            "selections": [
                {
                    "id": s["id"],
                    "category": s["category"],
                    "market": s["market"],
                    "selection": s["selection"],
                    "probability": s["probability"],
                    "odds": s.get("odds"),
                    "edge": s.get("edge"),
                    "top_pick": s["id"] == pick_id,
                    "best": s["id"] in best_ids,
                    "settle": _jsonable_rule(s["settle"]),
                    "won": None,
                }
                for s in report.selected
            ],
            "status": "pending",
            "result": None,
            "settled_at": None,
        }
        pair = {pred.player_a, pred.player_b}

        def same_match(r: dict[str, Any]) -> bool:
            return r["key"] == key or (
                r["match_date"] == match_date and {r["player_a"]["id"], r["player_b"]["id"]} == pair
            )

        existing = self.load()
        done = [r for r in existing if same_match(r) and r["status"] != "pending"]
        if done:
            return done[0]  # ya liquidado: no se reescribe
        records = [r for r in existing if not same_match(r)]
        records.append(record)
        self.write(records)
        return record

    # ------------------------------------------------------------------- liquidar
    def settle_pending(self, client: Any, today: str | None = None) -> dict[str, int]:
        """Busca en api-tennis.com los resultados de las predicciones pendientes."""
        records = self.load()
        pending = [r for r in records if r["status"] == "pending" and r["match_date"] < (today or "9999")]
        summary = {"pending": len(pending), "settled": 0, "void": 0, "not_found": 0}
        if not pending:
            return summary
        by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in pending:
            by_date[record["match_date"]].append(record)
        for match_date, items in sorted(by_date.items()):
            day = datetime.strptime(match_date, "%Y-%m-%d").date()
            # Un día de margen: el partido puede jugarse tarde o pasar al día siguiente.
            rows = client.raw_fixtures(day.isoformat(), (day + timedelta(days=1)).isoformat())
            for record in items:
                row = _find_row(record, rows)
                if row is None:
                    summary["not_found"] += 1
                    continue
                outcome = outcome_from_api(row, record)
                if outcome is None:
                    continue  # todavía no terminado
                if outcome == "void":
                    record["status"] = "void"
                    record["result"] = {"status": row.get("event_status")}
                    summary["void"] += 1
                else:
                    for sel in record["selections"]:
                        rule = _rule_from_json(sel["settle"])
                        sel["won"] = settle({"settle": rule}, outcome)
                    record["status"] = "settled"
                    record["result"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in outcome.items()}
                    summary["settled"] += 1
                record["settled_at"] = _now()
        self.write(records)
        return summary

    # -------------------------------------------------------------------- informe
    def report(self) -> str:
        records = self.load()
        settled = [r for r in records if r["status"] == "settled"]
        sels = [s for r in settled for s in r["selections"] if s["won"] is not None]
        lines = [
            "# 📒 Histórico de predicciones — Tenis-AI",
            f"- Partidos registrados: {len(records)} · comprobados: {len(settled)} · "
            f"pendientes: {sum(r['status'] == 'pending' for r in records)} · "
            f"nulos (retiro/walkover): {sum(r['status'] == 'void' for r in records)}",
        ]
        if not sels:
            lines.append("- Aún no hay selecciones comprobadas con resultados reales.")
            return "\n".join(lines)

        def block(items: list[dict[str, Any]]) -> str:
            hits = sum(1 for s in items if s["won"])
            mean_p = sum(s["probability"] for s in items) / len(items)
            text = f"{hits}/{len(items)} aciertos ({hits / len(items) * 100:.1f}%) · prob. media {mean_p * 100:.1f}%"
            priced = [s for s in items if s.get("odds")]
            if priced:
                pnl = sum((s["odds"] - 1) if s["won"] else -1 for s in priced)
                text += f" · ROI con cuota {pnl / len(priced) * 100:+.1f}% ({len(priced)} con cuota)"
            return text

        lines += ["", f"**Todas las selecciones:** {block(sels)}"]
        picks = [s for s in sels if s.get("top_pick")]
        if picks:
            lines.append(f"**Picks recomendados:** {block(picks)}")
        best = [s for s in sels if s.get("best")]
        if best:
            lines.append(f"**Mejores predicciones:** {block(best)}")
        lines += ["", "**Por categoría**"]
        by_cat: dict[str, list] = defaultdict(list)
        for s in sels:
            by_cat[s["category"]].append(s)
        lines += [f"- {cat}: {block(items)}" for cat, items in sorted(by_cat.items())]
        lines += ["", "**Calibración (lo que el modelo dijo frente a lo que pasó)**"]
        for lo, hi in SHOWN_BINS:
            items = [s for s in sels if lo <= s["probability"] < hi]
            if items:
                lines.append(f"- {lo * 100:.0f}–{min(hi, 0.88) * 100:.0f}%: {block(items)}")
        lines += ["", "**Últimos partidos comprobados**"]
        for r in settled[-10:]:
            res = r["result"] or {}
            ok = sum(1 for s in r["selections"] if s["won"])
            n = sum(1 for s in r["selections"] if s["won"] is not None)
            score = res.get("score_text", "")
            lines.append(
                f"- {r['match_date']} {r['player_a']['name']} vs {r['player_b']['name']} ({score}): {ok}/{n} aciertos"
            )
        return "\n".join(lines)


def _surname(name: str) -> str:
    parts = normalize(name).split()
    return parts[-1] if parts else ""


def _find_row(record: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if record.get("event_key"):
        for row in rows:
            if str(row.get("event_key")) == str(record["event_key"]):
                return row
    a, b = _surname(record["player_a"]["name"]), _surname(record["player_b"]["name"])
    for row in rows:
        if str(row.get("event_type_type") or "").lower() != "atp singles":
            continue
        first, second = _surname(str(row.get("event_first_player"))), _surname(str(row.get("event_second_player")))
        if {first, second} == {a, b}:
            return row
    return None


def outcome_from_api(row: dict[str, Any], record: dict[str, Any]) -> dict[str, Any] | str | None:
    """Resultado real (perspectiva A/B del registro) a partir de un evento de api-tennis.com.

    Devuelve None si aún no terminó y "void" si hubo retiro, walkover o cancelación.
    Los quiebres no se pueden comprobar con esta fuente (no trae estadísticas de saque).
    """
    status = str(row.get("event_status") or "")
    if status in ("Retired", "Walk Over", "Cancelled", "Postponed", "Abandoned"):
        return "void"
    if status != "Finished":
        return None
    a_is_first = _surname(record["player_a"]["name"]) == _surname(str(row.get("event_first_player")))
    sets: list[tuple[int, int]] = []
    for score in sorted(row.get("scores") or [], key=lambda s: int(float(s.get("score_set") or 0))):
        try:
            g1, g2 = int(float(score.get("score_first"))), int(float(score.get("score_second")))
        except (TypeError, ValueError):
            return None
        sets.append((g1, g2) if a_is_first else (g2, g1))
    if not sets:
        return None
    sets_a = sum(1 for x, y in sets if x > y)
    sets_b = len(sets) - sets_a
    winner_first = str(row.get("event_winner") or "") == "First Player"
    winner = "a" if winner_first == a_is_first else "b"
    outcome = {
        "winner": winner,
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
        "games_supported": all(max(x, y) <= 7 for x, y in sets),
        "score_text": " ".join(f"{x}-{y}" for x, y in sets),
    }
    outcome["total_games"] = outcome["games_a"] + outcome["games_b"]
    return outcome
