"""Parámetros del modelo ajustables con datos (patrón tomado de Deport-AI)."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

DEFAULT_PARAMS_PATH = Path(__file__).resolve().parents[2] / "model_params.json"


@dataclass(frozen=True)
class TennisParams:
    # Elo: K = elo_k_scale * 250 / (n + 5) ** 0.4 (n = partidos jugados)
    elo_k_scale: float = 1.0
    # Mezcla de rating global y de superficie (0 = solo global, 1 = solo superficie)
    elo_surface_weight: float = 0.5
    # Peso del Elo frente al modelo saque/resto en la probabilidad final (escala logit)
    elo_weight: float = 0.6
    # Saque/resto: vida media (días) de la ponderación temporal
    half_life_days: float = 365.0
    # Peso de los partidos en otra superficie al estimar saque/resto
    cross_surface_weight: float = 0.35
    # Puntos ficticios de contracción hacia la media del circuito
    prior_points: float = 600.0
    # Desviación de la "forma del día": variación aleatoria de la calidad de saque en
    # cada partido. Hace los partidos más desiguales que el modelo de puntos independientes
    # (menos juegos y tie-breaks) sin cambiar la probabilidad de victoria.
    form_sd: float = 0.035
    # Metadatos
    fitted: bool = False
    sample_size: int = 0
    log_loss: float | None = None
    baseline_log_loss: float | None = None
    fitted_at: str | None = None

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> "TennisParams":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


DEFAULT_PARAMS = TennisParams()

SEARCH_GRID: dict[str, list[float]] = {
    "elo_k_scale": [0.7, 0.85, 1.0, 1.2, 1.4],
    "elo_surface_weight": [0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0],
    "elo_weight": [0.0, 0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "half_life_days": [120.0, 240.0, 365.0, 540.0, 730.0],
    "cross_surface_weight": [0.1, 0.25, 0.35, 0.5, 0.75, 1.0],
    "prior_points": [200.0, 400.0, 600.0, 1000.0, 1600.0],
}

# form_sd no influye en el ganador: se ajusta aparte con el log-loss de los juegos totales.
FORM_SD_GRID = [0.0, 0.02, 0.03, 0.035, 0.04, 0.05, 0.06]


def params_path() -> Path:
    env = os.getenv("TENIS_AI_PARAMS")
    return Path(env) if env else DEFAULT_PARAMS_PATH


def load_params(path: Path | None = None) -> TennisParams:
    target = path or params_path()
    if not target.exists():
        return DEFAULT_PARAMS
    try:
        return TennisParams.from_json(json.loads(target.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        return DEFAULT_PARAMS


def save_params(params: TennisParams, path: Path | None = None) -> Path:
    target = path or params_path()
    target.write_text(json.dumps(params.to_json(), indent=2), encoding="utf-8")
    return target


def fit_params(
    objective: Callable[[TennisParams], tuple[float, int]],
    start: TennisParams = DEFAULT_PARAMS,
    grid: dict[str, list[float]] | None = None,
    passes: int = 2,
    log: Callable[[str], None] = lambda _m: None,
) -> TennisParams:
    """Descenso por coordenadas sobre la rejilla minimizando el log-loss walk-forward."""
    grid = grid or SEARCH_GRID
    baseline, n = objective(start)
    if not math.isfinite(baseline) or n == 0:
        return start
    best, best_loss = start, baseline
    log(f"  inicio: log-loss {baseline:.5f} ({n} partidos)")
    for _ in range(passes):
        improved = False
        for name, values in grid.items():
            for value in values:
                if getattr(best, name) == value:
                    continue
                candidate = replace(best, **{name: value})
                loss, _ = objective(candidate)
                if loss < best_loss - 1e-6:
                    best, best_loss, improved = candidate, loss, True
                    log(f"  {name}={value}: log-loss {loss:.5f}")
        if not improved:
            break
    return replace(
        best,
        fitted=True,
        sample_size=n,
        log_loss=round(best_loss, 5),
        baseline_log_loss=round(baseline, 5),
    )
