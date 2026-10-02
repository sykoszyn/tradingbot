# Trading bot — Especificación, arquitectura y plan (para aprobar)

Estado: **aprobado** (2026-10-02). Aprobación manual: operaciones de más de **US$1.000**. Límites de riesgo: los propuestos en §1.4. Deploy: **VPS**.

Decisiones ya tomadas:

- **Mercado:** Alpaca *paper*, acciones y ETFs de EE.UU. (universo inicial: SPY y QQQ).
- **Estrategia:** Opus 5.5 propone varias candidatas y el backtest decide.
- **Capa rápida:** un modelo calibrado propio con la misma interfaz que tendría Jev, para poder enchufar Jev después.
- **Proyecto:** en un repo nuevo.
- **Proceso:** no se instala AgenKit. Sigo sus 6 fases a mano (brainstorm → arquitectura → plan → tests primero → revisión → entrega), con pausas para tu aprobación.

---

## 1. Especificación

### Qué hace
1. **Cada vela:** el código arma un *snapshot* numérico del mercado. El modelo rápido devuelve probabilidades para un conjunto fijo de preguntas, y el código decide si opera, cuánto y con qué stop. Todo queda registrado.
2. **Cada noche:** Opus 5.5 revisa la sesión (cada fill y cada error), propone mejoras a la estrategia y a las preguntas, y el arnés las backtestea. Solo pasa a operar lo que supera el mismo filtro.
3. **Siempre:**
   - Un dashboard muestra en vivo cada señal, su probabilidad, la confianza, la acción y el resultado.
   - Telegram avisa cada fill, error, escalamiento y disparo del kill switch.
   - Todos los días sale un reporte.

### Qué NO hace (límites duros, en código)
- Ningún modelo puede cambiar un límite de riesgo, el tamaño máximo ni el kill switch. Los modelos aconsejan y el código decide.
- No opera dinero real hasta que el chequeo final (§1.6) pase completo y vos lo actives a mano.
- No pide ni guarda contraseñas ni códigos 2FA. Las claves van solo en `.env`, nunca en el código ni en los logs (los logs pasan por un filtro que las tapa).
- Titulares, datos de mercado y logs se tratan como **datos**, nunca como instrucciones. Opus los lee como contenido no confiable y su salida se valida contra un esquema JSON fijo.

### 1.1 Filtro para aceptar una estrategia (fuera de muestra)
| Métrica | Umbral |
|---|---|
| Sharpe anualizado (retornos diarios) | > 1,5 |
| Caída máxima | < 15% |
| Hit rate (operaciones ganadoras) | > 55% |
| t-stat del retorno medio por operación | > 2,0 |
| Datos | ≥ 2 años, varios regímenes (incluye 2020 y 2022), con costos y slippage |

Expectativa honesta: la mayoría de las estrategias no pasa este filtro. Si ninguna lo pasa, el bot **no opera**, y es el resultado correcto.

### 1.2 Preguntas de resultado fijo (capa rápida)
Una sola llamada por vela devuelve todas:

| ID | Pregunta | Tipo |
|---|---|---|
| `regime` | Régimen actual | elección: tendencia / rango / alta volatilidad |
| `direction` | Dirección en las próximas N velas | elección: sube / baja / lateral |
| `pressure_real` | ¿La presión compradora es real? | sí / no |
| `setup_quality` | Calidad del setup | puntaje 0–1 (en bins calibrados) |
| `risk_state` | Estado de riesgo | elección: calma / elevado / estrés |

- Cada pregunta mide un solo factor. El código las combina con pesos explícitos (definidos en `strategy.md`).
- Solo se opera si **cada** probabilidad supera su umbral.

### 1.3 Tamaño de posición
- **Fórmula:** f = min(¼ · Kelly, tope), con Kelly = p − (1 − p)/b, donde p es la probabilidad calibrada y b = ganancia media / pérdida media del setup en el backtest. Es 0 si p está por debajo del umbral.
- **Hasta que la calibración esté verificada con tus propios fills** (§1.5), el tamaño queda en el mínimo: una posición fija chica, no Kelly.

### 1.4 Reglas de riesgo (valores por defecto, a confirmar)
| Regla | Valor propuesto |
|---|---|
| Tamaño máximo por posición | 10% del capital |
| Exposición total máxima | 50% del capital |
| Pérdida diaria máxima | 2% → no opera más ese día |
| Caída máxima desde el pico | 10% → **kill switch**: cierra todo y frena hasta que lo reactives a mano |
| Aprobación manual | toda operación mayor a **US$1.000** (se pide por Telegram, con botones) |
| Operaciones por día | máximo 10 |
| Claves de Alpaca | solo permiso de trading; la API de trading no puede retirar fondos |

Las reglas se chequean **antes de cada orden**. La única forma de mandar una orden es a través de `risk.check()`, y un test garantiza que no existe otro camino.

### 1.5 Calibración
- Cada decisión se registra con su resultado.
- Por pregunta se mide el Brier score, la curva de confiabilidad (10 bins) y el ECE.
- Si la curva se tuerce, se recalibra en código (isotónica) usando los fills reales.
- Para usar Kelly hacen falta al menos 200 decisiones por pregunta y un ECE menor a 0,05.

### 1.6 Chequeo final antes de dinero real
El modo real se niega a arrancar si alguna de estas respuestas no da bien:
1. ¿El paper coincide con el backtest? Con ≥ 100 operaciones, la diferencia tiene que estar dentro de la tolerancia.
2. ¿El kill switch se disparó en un test real en los últimos 7 días?
3. ¿Hay algún límite delegado a un modelo en vez de al código? Lo verifica un test estático.
4. ¿La confianza está calibrada con tus propios fills?
5. ¿Qué régimen de mercado rompería esto? Se responde con el backtest por régimen.

Termina con la sección **"WHAT COULD BLOW UP THIS ACCOUNT?"**.

### 1.7 Lo que vas a recibir
- Bot corriendo en paper y link al dashboard.
- `strategy.md` con entrada, salida, stop, take profit, timeframe y condición exacta de invalidación.
- Reporte diario con: operaciones, P&L, win rate, mayor pérdida, latencia media y costo por decisión del modelo rápido, costo de Opus por noche y puntaje de calibración.

---

## 2. Arquitectura

Son tres capas que nunca se superponen:

```
                 ┌─────────────────────────── NOCHE (offline) ───────────────────────────┐
                 │  OPUS 5.5 (cerebro lento)                                              │
                 │  lee ledger + fills (como datos) → propone candidatas en JSON con      │
                 │  esquema fijo → el ARNÉS corre backtests → filtro §1.1 → strategy.json │
                 └──────────────────────────────┬─────────────────────────────────────────┘
                                                │ solo si pasa el filtro
 Alpaca ── velas/quotes ──► STATE ENGINE ──► snapshot ──► SCORER (capa rápida) ──► probabilidades
 (paper)                    (determinístico)              (modelo calibrado, <10 ms)        │
    ▲                                                                                       ▼
    │                       RIESGO + SIZING + KILL SWITCH (código, límites duros) ◄── DECISIÓN (código)
    └──────── órdenes ◄─────────────────── EJECUCIÓN (idempotente) ──► LEDGER (SQLite)
                                                                         │
                                                  DASHBOARD (vivo) · TELEGRAM · REPORTE DIARIO
```

### Componentes
| Módulo | Qué hace | Tecnología |
|---|---|---|
| `data/` | Descarga y cachea velas históricas y en vivo | `alpaca-py`, Parquet |
| `state/` | Snapshot: precio, spread, desequilibrio del book, volatilidad realizada, tendencia, flujo de órdenes. **Solo datos con timestamp anterior a la decisión.** | numpy/pandas puros |
| `scorer/` | Interfaz `Scorer.score(snapshot) → {pregunta: probabilidades}`. Implementación: gradient boosting + calibración isotónica, entrenado walk-forward | scikit-learn |
| `strategy/` | Plantillas de reglas (tendencia, reversión, ruptura) parametrizadas desde `strategy.json`; `strategy.md` se genera desde ahí | Python |
| `risk/` | Límites, sizing ¼ Kelly, kill switch (automático, por archivo y por `/kill` en Telegram) | Python, sin dependencias de los modelos |
| `exec/` | Órdenes a Alpaca paper con `client_order_id` idempotente y conciliación con el broker | `alpaca-py` |
| `backtest/` | Simulador vela por vela que usa **el mismo** state engine, estrategia y riesgo que en vivo. Costos (fees SEC/TAF) y slippage (medio spread + k·volatilidad) | Python |
| `brain/` | Investigación nocturna con Opus 5.5 (`claude-opus-5-5`, effort alto, fallbacks del servidor activos). Salida validada con esquema; nunca ejecuta código que escribe el modelo | `anthropic` SDK |
| `ledger/` | Señales, probabilidades, decisiones, órdenes, fills y resultados | SQLite |
| `dashboard/` | Señales en vivo, P&L, calibración | FastAPI + SSE + HTML/JS liviano |
| `alerts/` | Bot de Telegram (BotFather): avisos, aprobaciones con botones, `/kill`, `/status` | Telegram Bot API |

### Decisiones importantes de arquitectura
1. **"Desequilibrio del book" con datos reales.** El plan de datos gratis de Alpaca solo da el mejor bid/ask en vivo, no la profundidad del libro, y el histórico de quotes es muy pesado. Si el modelo usara en vivo algo que el backtest no tiene, *el backtest mentiría*. Por eso:
   - El snapshot usa medidas de flujo que se pueden calcular igual en el pasado y en el presente: posición del cierre dentro de la vela y volumen con signo (regla de tick).
   - El desequilibrio bid/ask en vivo se muestra en el dashboard, pero no entra al modelo hasta que tengamos histórico propio grabado.
2. **Opus no ejecuta código propio sin revisión.** De noche propone cambios dentro de plantillas y parámetros con esquema fijo, y eso se backtestea y se aplica solo si pasa el filtro. Si propone código nuevo, va como PR para que lo apruebes; nunca se aplica solo.
3. **Rollback:**
   - Cada `strategy.json` se versiona y la anterior queda guardada; volver atrás es un comando.
   - El deploy usa imágenes con tag, así que se puede volver a la imagen anterior.
4. **Deploy:** **VPS con Docker** (recomendado; unos US$5–6 por mes) o Mac Mini con `launchd`, con reinicio automático. **Vercel no sirve**: no corre procesos 24/7, solo funciones que duran segundos. El dashboard sí podría ir en Vercel más adelante, leyendo del VPS.

---

## 3. Plan (cada módulo: test que falla primero → código → revisión contra esta spec → rollback)

| # | Fase | Test que tiene que fallar primero (y después pasar) |
|---|---|---|
| M0 | Esqueleto: `uv`, pytest, ruff, config, carga de `.env` con logs que tapan claves | Un log con la clave impresa sale tapado |
| M1 | Datos: descarga y caché de Alpaca, controles de punto en el tiempo | Pedir datos "al momento t" nunca devuelve velas posteriores |
| M2 | State engine | Si cambio velas *futuras*, el snapshot en t no cambia (sin lookahead) |
| M3 | Riesgo + kill switch | Cada límite bloquea; no existe otro camino para enviar órdenes; el kill switch cierra todo |
| M4 | Backtester | Con datos sintéticos de P&L conocido da el resultado exacto, con costos |
| M5 | Scorer calibrado | Latencia < 10 ms; probabilidades que suman 1; Brier y ECE calculados bien |
| M6 | Plantillas de estrategia + filtro walk-forward | Una estrategia "trampa" con lookahead es detectada y rechazada |
| M7 | Investigación con Opus 5.5 | Una salida inválida del modelo se rechaza; Opus nunca toca `risk/` |
| M8 | Ejecución en Alpaca paper | Órdenes duplicadas no se mandan dos veces; la conciliación detecta diferencias |
| M9 | Dashboard + Telegram | Cada fill y cada error genera un aviso; `/kill` frena el bot |
| M10 | Ciclo nocturno + reporte diario | Un cambio que no pasa el filtro no se aplica |
| M11 | Deploy + chequeo final | El modo real se niega a arrancar si falla un punto del §1.6 |

**Revisión:** al final de cada módulo hago una revisión adversarial contra esta spec y te muestro qué quedó.

---

## 4. Lo que necesito de vos para arrancar

1. **Aprobación** de la especificación (§1), la arquitectura (§2) y el plan (§3), o los cambios que quieras.
2. **Repo nuevo** en GitHub, por ejemplo `sykoszyn/trading-bot`. Te recomiendo que sea **privado**: tus otros repos son públicos.
3. **Números a confirmar:**
   - El monto a partir del cual pedir aprobación manual.
   - Los límites de riesgo del §1.4.
   - Dónde va a correr: VPS o Mac Mini.
4. **Claves** (más adelante; nunca las pegues en el chat, van directo al `.env` del servidor):
   - Alpaca paper (*key* y *secret*).
   - Anthropic (para el Opus nocturno; estimo entre US$1 y US$5 por noche, con tope configurable).
   - Token del bot de Telegram.
5. **Red:** esta sesión no llega a `*.alpaca.markets` ni a `api.telegram.org`. Puedo construir y testear todo con datos sintéticos, pero para los backtests con datos reales hay que habilitar esos dominios en el entorno o correrlos en tu servidor.
