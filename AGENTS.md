# Reglas de Pronósticos de Tenis (TENIS-AI, ATP)

Cuando el usuario pida analizar un partido de tenis o los partidos del día (por ejemplo: "Analiza Alcaraz vs Sinner", "¿quién gana hoy en el ATP?"):

## 🎯 Flujo obligatorio
1. Partido concreto:
   ```bash
   python predict.py "<Jugador A> vs <Jugador B>" --surface <Hard|Clay|Grass> [--best-of 5]
   ```
   Si el usuario no indica la superficie, pregúntala o dedúcela del torneo. Al mejor de 5 solo en Grand Slams.
2. Partidos del día (requiere `API_TENNIS_KEY`):
   ```bash
   python predict.py --today [--date AAAA-MM-DD]
   ```
3. **Muestra directamente el reporte del comando.** No inventes ni recalcules probabilidades, líneas ni cuotas.
4. Si el comando muestra "⚠️ Incidencias" o avisos de datos antiguos, inclúyelos al final.

## 📒 Histórico de predicciones (obligatorio)
- `predict.py` guarda automáticamente cada predicción en `history/predictions.jsonl` (no usar `--no-save` salvo pruebas).
- **Después de cada predicción**, subir el histórico: `python track.py sync` (commit y push a `main`). Las sesiones en la nube son temporales: lo que no se sube se pierde.
- Para comprobar con los resultados reales: `python track.py settle`, luego `python track.py report` y `python track.py sync`.
- Al empezar una sesión nueva, ejecutar primero `python track.py settle` para liquidar lo pendiente.

## ⛔ Reglas de presentación (las aplica el código)
1. Solo selecciones entre **60.0 % y 88.0 %**.
2. **Sin hándicaps** de juegos ni de sets.
3. **Una sola selección por mercado** (p. ej. una única línea de juegos totales) y **máximo 4 por categoría**, de mayor a menor probabilidad.
4. Con cuotas: **EV = probabilidad × cuota − 1**; se descarta lo que tenga EV < +3 % y se elige la de mayor EV. El Pick Recomendado es la de mayor EV.
5. Sin tablas, sin "mercados a evitar", sin variables internas del modelo, sin etiquetas de nivel.
6. Una justificación de **una sola línea** por selección, empezando con *Porque ...*.
7. Categorías vacías no se muestran.
8. **Mejores predicciones del partido** (máx. 4): de lo ya seleccionado, solo los mercados más fiables del backtest (quiebres totales y de cada jugador, tie-break en el 1er set, juegos de cada jugador, del 1er set y totales), ordenados por probabilidad corregida con su acierto histórico (`market_reliability.json`). Sin cuotas, el Pick Recomendado es la primera de ellas.

## 🧪 Validación
- Cambios en el modelo: ejecutar `python -m pytest` y `python backtest.py` y comparar Brier/log-loss del ganador y la calibración de lo mostrado.
- La fiabilidad por mercado se recalcula con `python backtest.py --save-reliability`.
- Los parámetros se ajustan con `python backtest.py --fit --save-params` (ajuste 2016–2023, evaluación fuera de muestra 2024+).

## 🚀 Integración
- Al terminar cualquier cambio de código, después de que pasen `python -m pytest` y las comprobaciones, **integrar siempre en `main`** (merge de la rama de trabajo y `git push origin main`).

## 🚀 Integración en main (instrucción del usuario)
- Todo cambio de código, una vez que pasan `pytest` y (si toca el modelo) `python backtest.py`, se integra **siempre** en `main` y se sube (`git push origin main`). Si el push es rechazado: `git pull --rebase origin main` y repetir.
