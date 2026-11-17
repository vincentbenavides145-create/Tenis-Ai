"""Conector de api-tennis.com: calendario del día y cuotas pre-partido (ATP individual).

La clave se lee de la variable de entorno API_TENNIS_KEY (también se acepta la
variante API_TENIS_KEY); api-tennis.com la recibe en la URL como `APIkey`. Formatos de respuesta según la documentación pública;
el parseo es defensivo y las incidencias se informan en lugar de inventar datos.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import requests

BASE_URL = "https://api.api-tennis.com/tennis/"
KEY_ENV_VARS = ("API_TENNIS_KEY", "API_TENIS_KEY")
# Estados de api-tennis.com de un partido que todavía no ha empezado.
PENDING_STATUSES = ("", "not started", "ns", "scheduled")


def read_api_key() -> str:
    for name in KEY_ENV_VARS:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def _api_error_message(payload: Any) -> str:
    """Texto legible de un error de api-tennis.com.

    La API responde HTTP 200 con {"error": "1", "result": [{"msg": ..., "cod": ...}]}.
    """
    if not isinstance(payload, dict):
        return str(payload)[:200]
    messages = []
    result = payload.get("result")
    if isinstance(result, list):
        for item in result:
            if isinstance(item, dict) and item.get("msg"):
                text = str(item["msg"]) + (f" (código {item['cod']})" if item.get("cod") is not None else "")
                if text not in messages:
                    messages.append(text)
    for key in ("message", "error"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip() and value.strip() not in ("0", "1") and value not in messages:
            messages.append(value.strip())
    return "; ".join(messages) or "respuesta sin detalle del error"


class ApiTennisError(RuntimeError):
    pass


@dataclass
class Fixture:
    event_key: str
    date: str
    time: str
    player_a: str
    player_b: str
    tournament: str
    round: str
    event_type: str
    status: str
    odds: dict[str, float] = field(default_factory=dict)

    @property
    def is_pending(self) -> bool:
        """True si el partido aún no ha empezado (ni en juego, ni terminado, ni cancelado)."""
        return self.status.strip().lower() in PENDING_STATUSES


class ApiTennisClient:
    def __init__(self, api_key: str | None = None, timeout: float = 30.0) -> None:
        self.api_key = api_key or read_api_key()
        if not self.api_key:
            raise ApiTennisError("Falta API_TENNIS_KEY (variable de entorno con la clave de api-tennis.com).")
        self.timeout = timeout

    def _get(self, method: str, **params: Any) -> Any:
        query = {"method": method, "APIkey": self.api_key, **params}
        try:
            response = requests.get(BASE_URL, params=query, timeout=self.timeout)
        except requests.RequestException as exc:
            raise ApiTennisError(f"No se pudo conectar con api-tennis.com: {exc}") from exc
        try:
            payload = response.json()
        except ValueError as exc:
            raise ApiTennisError(f"Respuesta no válida de api-tennis.com (HTTP {response.status_code}).") from exc
        if (
            response.status_code >= 400
            or not isinstance(payload, dict)
            or str(payload.get("success")) not in ("1", "True", "true")
        ):
            raise ApiTennisError(
                f"api-tennis.com rechazó '{method}' (HTTP {response.status_code}): {_api_error_message(payload)}"
            )
        return payload.get("result")

    def atp_singles_fixtures(self, day: str) -> list[Fixture]:
        """Partidos ATP individuales (no dobles, no challengers) de la fecha AAAA-MM-DD."""
        rows = self._get("get_fixtures", date_start=day, date_stop=day) or []
        fixtures: list[Fixture] = []
        for row in rows:
            event_type = str(row.get("event_type_type") or "")
            if event_type.lower() != "atp singles":
                continue
            fixtures.append(
                Fixture(
                    event_key=str(row.get("event_key")),
                    date=str(row.get("event_date") or day),
                    time=str(row.get("event_time") or ""),
                    player_a=str(row.get("event_first_player") or ""),
                    player_b=str(row.get("event_second_player") or ""),
                    tournament=str(row.get("tournament_name") or ""),
                    round=str(row.get("tournament_round") or ""),
                    event_type=event_type,
                    status=str(row.get("event_status") or ""),
                )
            )
        return fixtures

    def finished_atp_results(self, date_start: str, date_stop: str) -> list[dict[str, Any]]:
        """Resultados ATP individuales terminados entre dos fechas (para actualizar el Elo).

        Cada elemento: {'date', 'tournament', 'winner', 'loser', 'sets': [(g_ganador, g_perdedor), ...]}.
        """
        rows = self._get("get_fixtures", date_start=date_start, date_stop=date_stop) or []
        results: list[dict[str, Any]] = []
        for row in rows:
            if str(row.get("event_type_type") or "").lower() != "atp singles":
                continue
            if str(row.get("event_status") or "").lower() != "finished":
                continue
            winner_flag = str(row.get("event_winner") or "")
            if winner_flag not in ("First Player", "Second Player"):
                continue
            first, second = str(row.get("event_first_player") or ""), str(row.get("event_second_player") or "")
            first_won = winner_flag == "First Player"
            sets = []
            for score in row.get("scores") or []:
                try:
                    g1, g2 = int(float(score.get("score_first"))), int(float(score.get("score_second")))
                except (TypeError, ValueError):
                    continue
                sets.append((g1, g2) if first_won else (g2, g1))
            results.append(
                {
                    "date": str(row.get("event_date") or date_start),
                    "tournament": str(row.get("tournament_name") or ""),
                    "winner": first if first_won else second,
                    "loser": second if first_won else first,
                    "sets": sets,
                }
            )
        results.sort(key=lambda r: r["date"])
        return results

    def match_winner_odds(self, event_key: str) -> dict[str, float]:
        """Cuota mediana de ganador del partido: {'winner_a': x, 'winner_b': y}."""
        try:
            result = self._get("get_odds", match_key=event_key) or {}
        except ApiTennisError:
            return {}
        markets = result.get(str(event_key)) or result.get(event_key) or {}
        home_away = markets.get("Home/Away") or {}
        out: dict[str, float] = {}
        for side, key in (("Home", "winner_a"), ("Away", "winner_b")):
            prices = []
            for value in (home_away.get(side) or {}).values():
                try:
                    price = float(value)
                except (TypeError, ValueError):
                    continue
                if price > 1.0:
                    prices.append(price)
            if prices:
                prices.sort()
                out[key] = prices[len(prices) // 2]
        return out
