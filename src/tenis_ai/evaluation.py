"""Métricas sin estado para validar pronósticos con resultados ya conocidos.

Estas utilidades no guardan partidos ni predicciones. Quien ejecute una
evaluación proporciona las probabilidades y los desenlaces de la muestra.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Sequence


@dataclass(frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    mean_predicted: float | None
    observed_frequency: float | None


@dataclass(frozen=True)
class EvaluationResult:
    brier_score: float
    log_loss: float
    calibration: list[CalibrationBin]


@dataclass(frozen=True)
class MulticlassEvaluationResult:
    brier_score: float
    log_loss: float
    accuracy: float
    sample_size: int
    home_accuracy: float
    draw_accuracy: float
    away_accuracy: float


def evaluate_multiclass_1x2(
    probs: Sequence[tuple[float, float, float]],
    outcomes: Sequence[str],
) -> MulticlassEvaluationResult:
    """Evalúa predicciones multiclase (1X2) con Brier Score Multiclase y Log Loss."""
    n = len(probs)
    if n == 0 or len(outcomes) != n:
        raise ValueError("Longitudes incompatibles o vacías.")

    total_brier = 0.0
    total_log_loss = 0.0
    correct = 0
    epsilon = 1e-15

    for (p_h, p_d, p_a), out in zip(probs, outcomes):
        out_clean = out.upper().strip()
        o_h = 1.0 if out_clean in {"1", "H", "HOME", "W"} else 0.0
        o_d = 1.0 if out_clean in {"X", "D", "DRAW", "E"} else 0.0
        o_a = 1.0 if out_clean in {"2", "A", "AWAY", "L"} else 0.0

        brier = (p_h - o_h) ** 2 + (p_d - o_d) ** 2 + (p_a - o_a) ** 2
        total_brier += brier

        p_true = p_h if o_h == 1.0 else (p_d if o_d == 1.0 else p_a)
        total_log_loss += -log(max(epsilon, p_true))

        pred_class = "1" if (p_h >= p_d and p_h >= p_a) else ("X" if p_d >= p_a else "2")
        actual_class = "1" if o_h == 1.0 else ("X" if o_d == 1.0 else "2")
        if pred_class == actual_class:
            correct += 1

    h_total = sum(1 for o in outcomes if o.upper().strip() in {"1", "H", "HOME", "W"})
    d_total = sum(1 for o in outcomes if o.upper().strip() in {"X", "D", "DRAW", "E"})
    a_total = sum(1 for o in outcomes if o.upper().strip() in {"2", "A", "AWAY", "L"})

    h_corr = sum(1 for (p_h, p_d, p_a), o in zip(probs, outcomes) if o.upper().strip() in {"1", "H", "HOME", "W"} and p_h >= max(p_d, p_a))
    d_corr = sum(1 for (p_h, p_d, p_a), o in zip(probs, outcomes) if o.upper().strip() in {"X", "D", "DRAW", "E"} and p_d >= max(p_h, p_a))
    a_corr = sum(1 for (p_h, p_d, p_a), o in zip(probs, outcomes) if o.upper().strip() in {"2", "A", "AWAY", "L"} and p_a >= max(p_h, p_d))

    return MulticlassEvaluationResult(
        brier_score=round(total_brier / n, 4),
        log_loss=round(total_log_loss / n, 4),
        accuracy=round(correct / n, 4),
        sample_size=n,
        home_accuracy=round(h_corr / max(1, h_total), 4),
        draw_accuracy=round(d_corr / max(1, d_total), 4),
        away_accuracy=round(a_corr / max(1, a_total), 4),
    )


def evaluate_binary_predictions(
    probabilities: Sequence[float], outcomes: Sequence[int | bool], bins: int = 10
) -> EvaluationResult:
    """Evalúa un mercado binario: por ejemplo, over 2.5 o ambos marcan."""
    return EvaluationResult(
        brier_score=brier_score(probabilities, outcomes),
        log_loss=log_loss(probabilities, outcomes),
        calibration=calibration_bins(probabilities, outcomes, bins=bins),
    )


def brier_score(probabilities: Sequence[float], outcomes: Sequence[int | bool]) -> float:
    probabilities, outcomes = _validated_pairs(probabilities, outcomes)
    return sum((probability - outcome) ** 2 for probability, outcome in zip(probabilities, outcomes)) / len(
        probabilities
    )


def log_loss(probabilities: Sequence[float], outcomes: Sequence[int | bool]) -> float:
    probabilities, outcomes = _validated_pairs(probabilities, outcomes)
    epsilon = 1e-15
    return -sum(
        outcome * log(max(epsilon, probability))
        + (1 - outcome) * log(max(epsilon, 1.0 - probability))
        for probability, outcome in zip(probabilities, outcomes)
    ) / len(probabilities)


def calibration_bins(
    probabilities: Sequence[float],
    outcomes: Sequence[int | bool],
    bins: int | Sequence[float] = 10,
) -> list[CalibrationBin]:
    """Agrupa por tramos de probabilidad. `bins` es un número de tramos iguales o
    una lista ascendente de bordes (p. ej. [0.60, 0.65, ..., 0.88])."""
    probabilities, outcomes = _validated_pairs(probabilities, outcomes)
    if isinstance(bins, int):
        if bins < 2:
            raise ValueError("bins debe ser al menos 2.")
        edges = [index / bins for index in range(bins + 1)]
    else:
        edges = [float(edge) for edge in bins]
        if len(edges) < 2 or edges != sorted(edges):
            raise ValueError("Los bordes deben ser al menos dos y ascendentes.")

    count = len(edges) - 1
    grouped: list[list[tuple[float, int]]] = [[] for _ in range(count)]
    for probability, outcome in zip(probabilities, outcomes):
        if probability < edges[0] or probability > edges[-1]:
            continue
        index = count - 1
        for position in range(count):
            if probability < edges[position + 1]:
                index = position
                break
        grouped[index].append((probability, outcome))

    result: list[CalibrationBin] = []
    for index, values in enumerate(grouped):
        lower, upper = edges[index], edges[index + 1]
        if values:
            mean_predicted = sum(value[0] for value in values) / len(values)
            observed = sum(value[1] for value in values) / len(values)
        else:
            mean_predicted = observed = None
        result.append(CalibrationBin(lower, upper, len(values), mean_predicted, observed))
    return result


def _validated_pairs(
    probabilities: Sequence[float], outcomes: Sequence[int | bool]
) -> tuple[list[float], list[int]]:
    if len(probabilities) != len(outcomes) or not probabilities:
        raise ValueError("probabilities y outcomes deben tener la misma longitud y no estar vacíos.")
    checked_probabilities = [float(value) for value in probabilities]
    if any(value < 0.0 or value > 1.0 for value in checked_probabilities):
        raise ValueError("Cada probabilidad debe estar entre 0 y 1.")
    checked_outcomes = [int(value) for value in outcomes]
    if any(value not in {0, 1} for value in checked_outcomes):
        raise ValueError("Cada resultado debe ser 0 o 1.")
    return checked_probabilities, checked_outcomes
