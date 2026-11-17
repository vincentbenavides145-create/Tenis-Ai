"""Conector de api-tennis.com: calendario del día y cuotas pre-partido (ATP individual).

La clave se lee de la variable de entorno API_TENNIS_KEY (api-tennis.com la recibe
en la URL como `APIkey`). Formatos de respuesta según la documentación pública;
el parseo es defensivo y las incidencias se informan en lugar de inventar datos.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import requests

BASE_URL = "https://api.api-tennis.com/tennis/"


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
    live: bool = False
    odds: dict[str, float] = field(default_factory=dict)

    @property
    def started(self) -> bool:
        """True si el partido ya está en juego o terminado (las cuotas ya no son pre-partido)."""
        return self.live or self.status.strip() not in ("", "Not Started")


class ApiTennisClient:
    def __init__(self, api_key: str | None = None, timeout: float = 30.0) -> None:
        self.api_key = (
            api_key or os.getenv("API_TENNIS_KEY", "").strip() or os.getenv("API_TENIS_KEY", "").strip()
        )
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
        if response.status_code >= 400 or str(payload.get("success")) not in ("1", "True", "true"):
            # Formato de error real: {"error": "1", "result": [{"msg": "Wrong login credentials", ...}]}
            details = payload.get("result")
            if isinstance(details, list) and details and isinstance(details[0], dict):
                message = "; ".join(str(d.get("msg")) for d in details if d.get("msg"))
            else:
                message = payload.get("message") or payload.get("error") or "error desconocido"
            raise ApiTennisError(f"api-tennis.com rechazó '{method}': {message}")
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
                    live=str(row.get("event_live") or "0") == "1",
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
