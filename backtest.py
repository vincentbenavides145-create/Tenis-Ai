"""Backtest walk-forward y ajuste de parámetros de Tenis-AI.

    python backtest.py                                   # evalúa 2024-2026 con los parámetros guardados
    python backtest.py --fit --save-params               # ajusta con 2016-2023 y evalúa 2024-2026
    python backtest.py --eval-from 20250101 --output informe.md
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from tenis_ai.backtesting import games_objective, run_backtest, winner_objective  # noqa: E402
from tenis_ai.data.sackmann import load_matches  # noqa: E402
from tenis_ai.params import DEFAULT_PARAMS, FORM_SD_GRID, fit_params, load_params, save_params  # noqa: E402


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def format_report(summary: dict, params, eval_from: str, eval_to: str, fit_info: str = "") -> str:
    w, shown = summary.get("winner", {}), summary["shown"]
    lines = [
        "# 📊 Backtest walk-forward — Tenis-AI (ATP)",
        f"**Fecha:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Periodo evaluado:** {eval_from} – {eval_to} · **partidos:** {summary['evaluated_matches']}",
        "",
        "## Parámetros",
        f"- K Elo ×{params.elo_k_scale} · peso superficie {params.elo_surface_weight} · peso Elo {params.elo_weight}",
        f"- Vida media {params.half_life_days:.0f} días · peso otra superficie {params.cross_surface_weight} "
        f"· puntos de contracción {params.prior_points:.0f} · forma del día {params.form_sd}",
        f"- Ajustado con datos: {params.fitted}" + (f" · {fit_info}" if fit_info else ""),
        "",
        "## 1. Ganador del partido",
        f"- Brier: **{w.get('brier')}** · Log-loss: **{w.get('log_loss')}** · Acierto del favorito: **{_pct(w.get('accuracy', 0))}**",
        f"- Solo Elo: Brier {w.get('brier_elo_only')} · Solo saque/resto: Brier {w.get('brier_serve_return_only')}",
        "",
        "## 2. Lo que ve el usuario (60–88 %)",
    ]
    if shown.get("count"):
        lines += [
            f"- Selecciones: {shown['count']} · prob. media {_pct(shown['mean_predicted'])} · acierto real "
            f"{_pct(shown['hit_rate'])} · diferencia **{shown['gap'] * 100:+.1f} pts** · Brier {shown['brier']}",
            "",
            "| Tramo | Muestra | Prob. media | Acierto real |",
            "|---|:---:|:---:|:---:|",
        ]
        lines += [f"| {c['bin']} | {c['count']} | {_pct(c['predicted'])} | {_pct(c['observed'])} |" for c in shown["calibration"]]
        lines += ["", "| Mercado | Muestra | Prob. media | Acierto real | Diferencia |", "|---|:---:|:---:|:---:|:---:|"]
        lines += [
            f"| {g} | {b['count']} | {_pct(b['mean_predicted'])} | {_pct(b['hit_rate'])} | {b['gap'] * 100:+.1f} |"
            for g, b in shown["by_group"].items()
        ]
    lines += ["", "## 3. Todos los mercados (calibración global)", "| Tramo | Muestra | Prob. media | Acierto real |", "|---|:---:|:---:|:---:|"]
    lines += [f"| {c['bin']} | {c['count']} | {_pct(c['predicted'])} | {_pct(c['observed'])} |" for c in summary["all"].get("calibration", [])]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest walk-forward y ajuste de Tenis-AI")
    parser.add_argument("--first-year", type=int, default=2010, help="Primer año cargado (calentamiento del Elo)")
    parser.add_argument("--fit", action="store_true", help="Ajusta los parámetros por máxima verosimilitud")
    parser.add_argument("--fit-from", default="20160101")
    parser.add_argument("--fit-to", default="20231231")
    parser.add_argument("--eval-from", default="20240101")
    parser.add_argument("--eval-to", default="20991231")
    parser.add_argument("--save-params", action="store_true", help="Guarda los parámetros ajustados en model_params.json")
    parser.add_argument("--output", default="", help="Guarda el informe en Markdown")
    parser.add_argument("--json", default="", help="Guarda el resumen en JSON")
    args = parser.parse_args()

    matches = load_matches(range(args.first_year, datetime.now().year + 1))
    print(f"Partidos cargados: {len(matches)} (hasta {matches[-1].date})")
    params, fit_info = (DEFAULT_PARAMS if args.fit else load_params()), ""
    if args.fit:
        print(f"Ajustando parámetros del ganador con {args.fit_from}–{args.fit_to}...")
        objective = winner_objective(matches, args.fit_from, args.fit_to)
        params = fit_params(objective, start=params, log=print)
        print("Ajustando la forma del día con el total de juegos...")
        games_obj = games_objective(matches, params, args.fit_from, args.fit_to)
        best_sd, best_loss = params.form_sd, games_obj(params.form_sd)[0]
        for sd in FORM_SD_GRID:
            loss, _ = games_obj(sd)
            print(f"  form_sd={sd}: log-loss juegos {loss:.4f}")
            if loss < best_loss:
                best_sd, best_loss = sd, loss
        params = replace(params, form_sd=best_sd, fitted_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        fit_info = f"log-loss ganador {params.baseline_log_loss} → {params.log_loss} ({args.fit_from}–{args.fit_to})"
        if args.save_params:
            print(f"Parámetros guardados en {save_params(params)}")

    print(f"Backtest {args.eval_from}–{args.eval_to} (todos los mercados)...")
    result = run_backtest(matches, params, args.eval_from, args.eval_to, progress=lambda n: print(f"  {n} partidos"))
    summary = result.summary()
    report = format_report(summary, params, args.eval_from, min(args.eval_to, matches[-1].date), fit_info)
    print("\n" + report)
    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
    if args.json:
        Path(args.json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
