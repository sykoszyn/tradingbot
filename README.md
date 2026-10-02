# tradingbot

Bot de trading en tres capas: **Opus 5.5** (cerebro lento, investiga y revisa de noche), un **modelo rápido calibrado**
(puntúa cada vela en milisegundos) y **código determinístico** que es dueño del estado, los límites, el tamaño y las órdenes.
Los modelos aconsejan; el código decide.

Opera en **Alpaca paper** hasta que el chequeo final (ver `docs/SPEC.md` §1.6) pase completo.

Ver la especificación aprobada en [`docs/SPEC.md`](docs/SPEC.md).
