# Tenis-AI

Predicciones de partidos **ATP** (individual masculino): ganador, sets, juegos, tie-breaks y quiebres de saque, con el formato de reporte de [Deport-AI](https://github.com/vincentbenavides145-create/Deport-AI).

```text
Historial ATP (formato Sackmann)  +  resultados recientes (api-tennis.com)
            ↓
Elo global y por superficie  +  saque/resto ajustado por rival
            ↓
Mezcla de ambas probabilidades (peso ajustado con datos)
            ↓
Cadena de Markov exacta punto → juego → tie-break → set → partido
(con "forma del día" integrada por cuadratura)
            ↓
Mercados 60–88 %, una selección por mercado, EV con cuotas
```

## Instalación

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
cp .env.example .env    # y pon tu API_TENNIS_KEY (opcional, solo para --today)
```

En el entorno cloud de Claude Code, la clave va en **Variables de entorno** como `API_TENNIS_KEY=<clave>` (una línea, formato NOMBRE=VALOR). Hay que abrir una sesión nueva después de guardarla.

## Uso

```bash
python predict.py "Alcaraz vs Sinner" --surface Hard          # partido manual
python predict.py "Djokovic vs Zverev" --surface Clay --best-of 5
python predict.py --today                                      # calendario ATP del día (api-tennis.com)
python predict.py --today --date 2026-11-18
```

`--today` primero actualiza el Elo con los resultados ATP desde el final del histórico hasta ayer, luego trae los partidos del día y las cuotas de ganador (mediana de las casas), y genera un reporte por partido. La superficie se deduce del torneo y al mejor de 5 se aplica en Grand Slams. Las incidencias (jugadores sin historial, torneos sin superficie conocida) se listan al final.

## Histórico de predicciones

Cada predicción (manual o `--today`) se guarda automáticamente en `history/predictions.jsonl`: jugadores, fecha, torneo, probabilidad de victoria y todas las selecciones mostradas con su cuota y EV. Ese archivo está dentro del repositorio para que no se pierda (las sesiones en la nube son temporales).

```bash
python track.py settle   # busca en api-tennis.com el resultado real de cada predicción pendiente
python track.py report   # aciertos por categoría, calibración (dijo X % → pasó Y %), ROI con cuota
python track.py sync     # commit + push del histórico a main
```

Se comprueban ganador, sets, juegos y tie-breaks. Los quiebres quedan sin comprobar, porque api-tennis.com no trae estadísticas de saque. Los partidos con retiro o walkover cuentan como nulos. Si un partido se vuelve a predecir antes de empezar, se guarda la última versión; una vez comprobado ya no se modifica.

## Modelo

- **Elo** global y por superficie, K = 250/(n+5)^0.4, mezcla 50/50 global/superficie (ajustada).
- **Saque/resto**: puntos ganados al saque y al resto con vida media de 120 días, peso 0.5 para otras superficies y contracción de 400 puntos hacia la media del circuito; ajuste por rival: p_saque(A) = media + (saque_A − media) − (resto_B − media).
- **Mezcla**: 50 % Elo y 50 % saque/resto en escala logit (ajustado). Después se desplazan ambas probabilidades de saque con un mismo delta para que el modelo de puntos reproduzca esa probabilidad, de modo que **todos los mercados son coherentes con el ganador**.
- **Markov exacto**: distribución de juegos, quiebres, tie-breaks y marcador en sets, sin simulación. Set decisivo de Grand Slam con tie-break a 10.
- **Forma del día** (σ = 0.035): variación aleatoria de la calidad de saque en cada partido. Sin ella, el modelo de puntos independientes sobreestimaba los juegos (+1.6 al mejor de 3) y los tie-breaks.

Parámetros en `model_params.json`, ajustados por máxima verosimilitud walk-forward con 17.382 partidos (2016–2023).

## Validación (fuera de muestra, 2024 – mayo 2026, 5.853 partidos)

`python backtest.py` — cada partido se predice solo con partidos anteriores. Informe completo: [`backtest_report.md`](backtest_report.md).

- **Ganador del partido:** Brier **0.2135**, log-loss **0.614**, acierto del favorito **64.9 %**. La mezcla mejora al Elo solo (0.2177) y al saque/resto solo (0.2192).
- **Lo que ve el usuario (60–88 %):** 66.528 selecciones, probabilidad media **77.0 %** frente a acierto real **76.8 %** (−0.2 puntos). Por tramos:
  - 60–65 %: 62.4 % predicho / 63.6 % real
  - 70–75 %: 72.8 % / 73.2 %
  - 80–88 %: 83.9 % / 83.0 %
- Ningún mercado se desvía más de 3.1 puntos. El mayor sesgo es "gana al menos un set" del jugador con id menor (−3.1).

Sin cuotas históricas no se mide el ROI: estar calibrado no implica ganarle a la casa. Con cuotas, el reporte muestra el EV y descarta lo que tenga EV < +3 %.

## Datos y limitaciones

- **Historial**: partidos ATP en formato Jeff Sackmann (licencia **CC BY-NC-SA 4.0, uso no comercial**). El repositorio original de GitHub ya no está disponible; se usa el espejo de archivo de Hugging Face [`Aneeshers/tennis-sackmann-archive`](https://huggingface.co/datasets/Aneeshers/tennis-sackmann-archive), que llega hasta **mayo de 2026**. Se descarga y guarda en `data/raw/` la primera vez.
- Los resultados posteriores llegan desde api-tennis.com **solo para el Elo** (la API no trae estadísticas de saque), así que el componente saque/resto queda congelado en mayo de 2026 hasta que el espejo se actualice.
- Solo nivel ATP (sin challengers ni qualys): un jugador que viene de challengers tiene poco historial y el reporte lo marca con confianza baja.
- El conector de api-tennis.com sigue la documentación pública y está probado con respuestas simuladas; la primera ejecución real puede requerir ajustes.

## Estructura

```text
predict.py, backtest.py
src/tenis_ai/
├── data/sackmann.py     # descarga y parseo del historial
├── data/api_tennis.py   # calendario, cuotas y resultados recientes
├── ratings.py           # Elo por superficie + saque/resto
├── markov.py            # cadena de Markov exacta
├── model.py             # mezcla, alineación y forma del día
├── markets.py           # catálogo de mercados y liquidación
├── selection.py         # reglas 60–88 %, EV
├── formatter.py         # reporte Markdown
├── engine.py            # orquestación
├── backtesting.py       # walk-forward y objetivos de ajuste
├── params.py            # parámetros y ajuste por coordenadas
└── evaluation.py        # Brier, log-loss, calibración
```
