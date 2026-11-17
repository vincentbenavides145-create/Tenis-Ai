# 📊 Backtest walk-forward — Tenis-AI (ATP)
**Fecha:** 2026-11-17 13:48 UTC
**Periodo evaluado:** 20240101 – 20260525 · **partidos:** 5853

## Parámetros
- K Elo ×1.0 · peso superficie 0.5 · peso Elo 0.5
- Vida media 120 días · peso otra superficie 0.5 · puntos de contracción 400 · forma del día 0.035
- Ajustado con datos: True · log-loss ganador 0.60965 → 0.60813 (20160101–20231231)

## 1. Ganador del partido
- Brier: **0.2135** · Log-loss: **0.6138** · Acierto del favorito: **64.9%**
- Solo Elo: Brier 0.2177 · Solo saque/resto: Brier 0.2192

## 2. Lo que ve el usuario (60–88 %)
- Selecciones: 66528 · prob. media 77.0% · acierto real 76.8% · diferencia **-0.2 pts** · Brier 0.1738

| Tramo | Muestra | Prob. media | Acierto real |
|---|:---:|:---:|:---:|
| 60-65% | 6955 | 62.4% | 63.6% |
| 65-70% | 5904 | 67.5% | 68.4% |
| 70-75% | 9700 | 72.8% | 73.2% |
| 75-80% | 16433 | 77.6% | 77.2% |
| 80-88% | 27536 | 83.9% | 83.0% |

| Mercado | Muestra | Prob. media | Acierto real | Diferencia |
|---|:---:|:---:|:---:|:---:|
| a_wins_set | 3922 | 74.4% | 71.2% | -3.1 |
| b_wins_set | 3799 | 74.0% | 75.3% | +1.4 |
| breaks_a | 5710 | 81.3% | 80.2% | -1.1 |
| breaks_b | 5701 | 81.3% | 80.2% | -1.1 |
| first_set | 2881 | 68.4% | 68.9% | +0.5 |
| first_set_games | 5853 | 79.1% | 78.7% | -0.4 |
| first_set_tiebreak | 5735 | 79.8% | 79.7% | -0.1 |
| games_a | 5852 | 79.1% | 78.2% | -0.8 |
| games_b | 5853 | 78.9% | 79.8% | +0.9 |
| set_score | 404 | 68.0% | 70.5% | +2.6 |
| tiebreak | 2702 | 65.0% | 66.0% | +0.9 |
| total_breaks | 5747 | 82.6% | 81.3% | -1.3 |
| total_games | 5853 | 77.3% | 77.2% | -0.0 |
| total_sets | 3153 | 68.3% | 70.9% | +2.6 |
| winner | 3363 | 70.2% | 70.7% | +0.4 |

## 3. Todos los mercados (calibración global)
| Tramo | Muestra | Prob. media | Acierto real |
|---|:---:|:---:|:---:|
| 0-10% | 14373 | 6.4% | 6.6% |
| 10-20% | 42873 | 15.7% | 16.4% |
| 20-30% | 57671 | 24.8% | 25.2% |
| 30-40% | 59745 | 35.1% | 34.9% |
| 40-50% | 71106 | 45.0% | 44.9% |
| 50-60% | 71284 | 55.0% | 55.1% |
| 60-70% | 60418 | 64.9% | 65.1% |
| 70-80% | 53477 | 74.9% | 74.3% |
| 80-90% | 40843 | 84.5% | 83.4% |
| 90-100% | 17318 | 93.7% | 93.3% |