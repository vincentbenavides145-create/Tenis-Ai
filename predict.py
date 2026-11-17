"""Predicciones ATP de Tenis-AI.

    python predict.py "Alcaraz vs Sinner" --surface Hard            # partido manual
    python predict.py "Alcaraz vs Sinner" --surface Clay --best-of 5
    python predict.py --today                                       # calendario ATP de hoy (api-tennis.com)
    python predict.py --today --date 2026-11-17
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

try:  # .env opcional (API_TENNIS_KEY=...); las variables de entorno ya definidas tienen prioridad
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

from tenis_ai.engine import TennisEngine, parse_matchup  # noqa: E402
from tenis_ai.history import HistoryStore  # noqa: E402
from tenis_ai.players import PlayerNotFound  # noqa: E402

GRAND_SLAMS = ("australian open", "roland garros", "french open", "wimbledon", "us open")


def warn_data_age(engine: TennisEngine) -> None:
    """Avisa si los datos con los que se predice están desactualizados."""
    history_age = (date.today() - engine.history_end).days
    if engine.data_end > engine.history_end:
        if history_age > 45:
            print(
                f"ℹ️ Elo actualizado con api-tennis.com hasta el {engine.data_end.isoformat()}; las estadísticas "
                f"de saque/resto del histórico terminan el {engine.history_end.isoformat()} ({history_age} días).\n",
                file=sys.stderr,
            )
    elif engine.data_age_days > 45:
        print(
            f"⚠️ Los datos históricos terminan el {engine.data_end.isoformat()} "
            f"({engine.data_age_days} días). La forma reciente de los jugadores no está incluida.\n",
            file=sys.stderr,
        )


def run_today(engine: TennisEngine, day: str, default_surface: str, store: HistoryStore | None) -> int:
    from tenis_ai.data.api_tennis import ApiTennisClient, ApiTennisError

    errors: list[str] = []
    try:
        client = ApiTennisClient()
        if engine.data_age_days > 1:
            print(f"Actualizando Elo con resultados ATP desde {engine.data_end.isoformat()} (api-tennis.com)...", file=sys.stderr)
            added, issues = engine.update_from_api(client)
            print(f"  {added} partidos añadidos; {len(issues)} con jugadores sin historial.\n", file=sys.stderr)
        else:
            issues = []
        warn_data_age(engine)
        fixtures = client.atp_singles_fixtures(day)
    except ApiTennisError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    if issues:
        errors.append(f"{len(issues)} resultados recientes con jugadores sin historial ATP (no actualizan el Elo).")
    if not fixtures:
        print(f"No hay partidos ATP individuales el {day} en api-tennis.com.")
        return 0
    print(f"Partidos ATP del {day}: {len(fixtures)}\n")
    saved = 0
    for fx in fixtures:
        if fx.started:
            errors.append(
                f"{fx.player_a} vs {fx.player_b}: ya está en juego o terminado ({fx.status}); "
                "no se analiza porque la predicción y las cuotas serían de antes del partido."
            )
            continue
        surface = engine.guess_surface(fx.tournament)
        note = fx.tournament + (f" · {fx.round}" if fx.round else "") + (f" · {fx.time}" if fx.time else "")
        if surface is None:
            surface = default_surface
            errors.append(f"{fx.player_a} vs {fx.player_b}: superficie desconocida para '{fx.tournament}', se usó {surface}.")
        best_of = 5 if any(s in fx.tournament.lower() for s in GRAND_SLAMS) else 3
        odds = client.match_winner_odds(fx.event_key)
        try:
            report = engine.report(fx.player_a, fx.player_b, surface, best_of, odds=odds, header_note=note)
        except PlayerNotFound as exc:
            errors.append(f"{fx.player_a} vs {fx.player_b}: {exc}")
            continue
        print(report.markdown + "\n\n---\n")
        if store is not None:
            store.save_prediction(report, fx.date or day, fx.tournament, fx.event_key, source="today")
            saved += 1
    if store is not None and saved:
        print(f"📒 {saved} predicciones guardadas en el histórico ({store.path.name}).", file=sys.stderr)
    if errors:
        print("## ⚠️ Incidencias")
        for err in errors:
            print(f"- {err}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Predicciones ATP de Tenis-AI")
    parser.add_argument("query", nargs="*", help="'Jugador A vs Jugador B'")
    parser.add_argument("--surface", default="Hard", choices=["Hard", "Clay", "Grass"])
    parser.add_argument("--best-of", type=int, default=3, choices=[3, 5])
    parser.add_argument("--today", action="store_true", help="Analiza el calendario ATP del día (requiere API_TENNIS_KEY)")
    parser.add_argument("--date", default=date.today().isoformat(), help="Fecha del partido AAAA-MM-DD")
    parser.add_argument("--tournament", default="", help="Torneo (se guarda en el histórico)")
    parser.add_argument("--no-save", action="store_true", help="No guardar la predicción en el histórico")
    args = parser.parse_args()

    store = None if args.no_save else HistoryStore()
    engine = TennisEngine()
    if args.today:
        sys.exit(run_today(engine, args.date, args.surface, store))
    warn_data_age(engine)
    if not args.query:
        parser.error("Indica un partido ('Alcaraz vs Sinner') o usa --today.")
    try:
        a, b = parse_matchup(" ".join(args.query))
        report = engine.report(a, b, args.surface, args.best_of)
        print(report.markdown)
        if store is not None:
            store.save_prediction(report, args.date, args.tournament, source="manual")
            print(f"\n📒 Predicción guardada en el histórico ({store.path.name}).", file=sys.stderr)
    except (ValueError, PlayerNotFound) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
