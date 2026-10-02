# tradingbot

Bot de trading en tres capas que nunca se superponen:

| Capa | Qué hace | Cuándo |
|---|---|---|
| **Opus 5.5** (cerebro lento) | Diseña y revisa estrategias. Propone especificaciones JSON con esquema cerrado; no ejecuta código propio. | De noche |
| **Capa rápida calibrada** | Responde 5 preguntas fijas (régimen, dirección, presión real, calidad del setup, riesgo) con probabilidades calibradas, en < 1 ms. | Cada vela |
| **Código determinístico** | Es dueño del estado del mercado, cada umbral, cada tamaño, cada veto de riesgo y cada orden. | Siempre |

**Los modelos aconsejan; el código decide.** Opera en **Alpaca paper** hasta que el chequeo final pase completo.

Especificación aprobada: [`docs/SPEC.md`](docs/SPEC.md) · Deploy en VPS: [`deploy/README.md`](deploy/README.md)

## Cómo funciona

```
 Alpaca ─ velas ─► STATE ENGINE ─ snapshot ─► CAPA RÁPIDA ─ probabilidades ─► ESTRATEGIA (código)
                   (solo datos pasados)        (< 1 ms, calibrada)                    │
                                                                                       ▼
        Alpaca ◄── órdenes bracket ◄── ROUTER ◄── RIESGO + SIZING ¼ KELLY + KILL SWITCH
                                          │
                     LEDGER ─► DASHBOARD en vivo · TELEGRAM · REPORTE DIARIO
                        │
                        └─► de noche: OPUS 5.5 revisa → el ARNÉS backtestea → solo pasa lo que supera el filtro
```

- **Sin lookahead:** el snapshot usa ventanas fijas hacia atrás y es idéntico en vivo y en el backtest (está
  testeado). El filtro además recalcula las features cortando los datos en momentos al azar, y rechaza la
  estrategia si alguna feature cambia.
- **Backtest honesto:** usa el mismo código que en vivo (state engine, estrategia, sizing y motor de riesgo), con
  slippage y fees, entrada a la apertura siguiente, y el stop gana si en una vela se tocan el stop y el TP.
- **Filtro (fuera de muestra, walk-forward purgado):** Sharpe > 1,5 · caída máx. < 15% · hit rate > 55% ·
  t-stat > 2 · ≥ 2 años · ≥ 30 operaciones. Lo calcula el arnés, nunca el modelo. Está en `config/gate.toml`.
- **Riesgo en código** (`config/risk.toml`, montado de solo lectura en Docker):
  - máximo 10% del capital por posición y 50% de exposición total;
  - −2% en el día frena las entradas hasta el día siguiente;
  - −10% desde el pico dispara el **kill switch**: cierra todo y frena hasta reactivarlo a mano;
  - aprobación por Telegram en operaciones de más de US$1.000;
  - máximo 10 operaciones por día.

  Toda orden pasa por `OrderRouter` → `RiskEngine.check()`; un test verifica que no exista otro camino.
- **Tamaño:** ¼ de Kelly sobre la probabilidad calibrada, con topes. **Hasta que la calibración esté verificada con
  tus propios fills** (≥ 200 decisiones y ECE < 0,05), usa un tamaño mínimo fijo.
- **Datos como datos:** lo que lee Opus (fills, logs, titulares) va marcado como datos no confiables, y su respuesta
  se valida contra un esquema que no tiene ningún campo de riesgo.

## Uso rápido (local)

```bash
uv sync
uv run pytest
uv run tbot backtest --synthetic --template trend
uv run tbot run --sim --days 5                  # simulación completa sin Alpaca
DASHBOARD_PASSWORD=una-clave-larga uv run tbot dashboard --sim   # http://127.0.0.1:8080 (usuario admin)
```

Con claves en `.env` (ver `.env.example`):

```bash
uv run tbot research          # Opus 5.5 + arnés → strategy/strategy.md (o candidate.json si nada pasa)
uv run tbot run --auto        # opera en paper con la aprobada; si no hay, observa con la candidata
uv run tbot kill-drill        # simulacro real del kill switch (en horario de mercado)
uv run tbot golive-check      # chequeo final + "WHAT COULD BLOW UP THIS ACCOUNT?"
```

## Estructura

```
config/       risk.toml (límites duros) · gate.toml (filtro) · settings.toml · events.csv (eventos macro)
src/tbot/
  data/       velas de Alpaca con caché, vista punto-en-el-tiempo, datos sintéticos
  state/      snapshot numérico compacto
  scorer/     preguntas fijas, modelo calibrado, etiquetas, calibración con fills propios
  strategy/   especificación cerrada, plantillas, strategy.md
  risk/       límites, ¼ Kelly, kill switch
  exec/       router (único camino), Alpaca (brackets), broker simulado
  backtest/   motor, costos, métricas, walk-forward, filtro y detector de lookahead
  brain/      Opus 5.5: investigación y revisión nocturna
  live/       runner por vela y servicio 24/7
  dashboard/  FastAPI + SSE, con contraseña
  alerts/     Telegram (avisos, aprobaciones, /kill, /status)
  report/     reporte diario
  golive/     chequeo final antes de dinero real
strategy/     strategy.json · strategy.md · history/ · lessons.md (se generan)
var/          ledger SQLite, estado, modelos, reportes (no se suben)
```

## Sobre Jev y AgenKit

El prompt original pedía usar "Jev" como capa rápida y "AgenKit" como arnés.

- **Jev** no tiene una API pública verificable (en npm es un paquete vacío). Por eso la capa rápida es un modelo propio
  con la misma interfaz (`score(snapshot) → {pregunta: {clase: probabilidad}}`): si conseguís acceso a Jev, se
  reemplaza `scorer/model.py` sin tocar nada más.
- **AgenKit** no se instaló: es un paquete nuevo, de un solo autor, que modifica la configuración de agentes del
  proyecto. Sus seis fases se siguieron igual, con pausas de aprobación.

> Esto es software experimental. Que pase el backtest no garantiza ganancias: los mercados cambian.
