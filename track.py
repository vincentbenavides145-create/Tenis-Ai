"""Histórico de predicciones de Tenis-AI.

    python track.py settle    # comprueba las predicciones pendientes con los resultados reales (api-tennis.com)
    python track.py report    # aciertos, calibración y ROI de lo predicho
    python track.py sync      # sube el histórico al repositorio (git commit + push a main)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from tenis_ai.history import HistoryStore  # noqa: E402


def cmd_settle(store: HistoryStore) -> int:
    from tenis_ai.data.api_tennis import ApiTennisClient, ApiTennisError

    try:
        summary = store.settle_pending(ApiTennisClient(), today=date.today().isoformat())
    except ApiTennisError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(
        f"Pendientes revisadas: {summary['pending']} · comprobadas: {summary['settled']} · "
        f"nulas (retiro/walkover): {summary['void']} · sin encontrar en la API: {summary['not_found']}"
    )
    return 0


def cmd_sync(store: HistoryStore) -> int:
    repo = Path(__file__).resolve().parent
    rel = store.path.relative_to(repo)
    subprocess.run(["git", "add", str(rel)], cwd=repo, check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=repo).returncode == 0:
        print("El histórico ya está al día en git.")
        return 0
    subprocess.run(["git", "commit", "-m", f"Histórico de predicciones {date.today().isoformat()}"], cwd=repo, check=True)
    push = subprocess.run(["git", "push", "origin", "HEAD:main"], cwd=repo)
    return push.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Histórico de predicciones de Tenis-AI")
    parser.add_argument("command", choices=["settle", "report", "sync"])
    args = parser.parse_args()
    store = HistoryStore()
    if args.command == "settle":
        sys.exit(cmd_settle(store))
    if args.command == "report":
        print(store.report())
        return
    sys.exit(cmd_sync(store))


if __name__ == "__main__":
    main()
