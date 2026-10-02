# Deploy en un VPS (24/7, con reinicio automático)

Costo aproximado: **US$5–6 por mes** (Hetzner CX22, DigitalOcean Basic de 1 GB o similar) más Opus de noche
(tope configurable, por defecto US$5 por noche).

> Vercel no sirve para esto: corre funciones que duran segundos, no un proceso que esté prendido todo el día.

## 1. Cuentas y claves (una sola vez)

1. **Alpaca paper:** creá la cuenta en alpaca.markets → *Paper Trading* → *API Keys* → *Generate*. Copiá la key y el
   secret. Arranca con US$100.000 de mentira.
2. **Telegram:** en Telegram hablale a **@BotFather** → `/newbot` → copiá el token. Después mandale `/start` a tu bot.
3. **Anthropic (Opus 5.5):** en console.anthropic.com → *API Keys*. En *Limits* poné un tope mensual de gasto.
4. **Contraseña del dashboard:** inventá una de al menos 10 caracteres.

**Nunca pegues estas claves en un chat ni las subas a GitHub.** Van solo en el archivo `.env` del servidor.

## 2. Servidor

1. Creá un VPS con **Ubuntu 24.04** y entrá por SSH con tu clave: `ssh root@IP_DEL_SERVIDOR`.
2. Firewall: solo SSH abierto (el dashboard no se expone a internet).
   ```bash
   ufw allow OpenSSH && ufw --force enable
   ```
3. Docker:
   ```bash
   curl -fsSL https://get.docker.com | sh
   ```
4. Código:
   ```bash
   git clone https://github.com/sykoszyn/tradingbot.git && cd tradingbot
   cp .env.example .env && nano .env      # completá las claves
   mkdir -p var strategy && chown -R 1000:1000 var strategy
   docker compose build
   ```
5. Tu chat de Telegram (después de mandarle `/start` al bot):
   ```bash
   docker compose run --rm bot tbot telegram-id   # copiá TELEGRAM_CHAT_ID al .env
   ```

## 3. Investigar la estrategia

```bash
docker compose run --rm bot tbot backtest --template trend   # prueba rápida con datos reales de Alpaca
docker compose run --rm bot tbot research                    # Opus 5.5 propone, el arnés evalúa (tarda varios minutos)
```

- **Si alguna pasa el filtro:** queda en `strategy/strategy.json` y `strategy/strategy.md`, y el bot opera en paper.
- **Si ninguna pasa (lo más probable al principio):** la mejor queda en `strategy/candidate.json`. El bot corre en
  **modo observación**: registra cada predicción y su resultado sin operar, y así se mide la calibración con datos
  reales. Podés volver a correr `tbot research` cuando quieras.

## 4. Prender el bot

```bash
docker compose up -d
docker compose logs -f bot     # ver qué hace (Ctrl+C para salir; el bot sigue)
```

- Se reinicia solo si se cae o si deja de responder (healthcheck + autoheal).
- Al arrancar te avisa por Telegram ("▶️ Bot iniciado"). Si ves ese mensaje sin haberlo reiniciado vos, es que se
  reinició solo: revisá los logs.
- Los stops y take profits quedan **en Alpaca** (órdenes bracket): protegen la posición aunque el bot esté caído.

## 5. Dashboard

Por seguridad solo escucha dentro del servidor. Desde tu computadora abrí un túnel SSH:

```bash
ssh -N -L 8080:127.0.0.1:8080 root@IP_DEL_SERVIDOR
```

y entrá a **http://localhost:8080** (usuario `admin`, la contraseña de `DASHBOARD_PASSWORD`).

Si querés entrar desde el celular sin túnel, poné un dominio con HTTPS delante (por ejemplo Caddy o Cloudflare
Tunnel). No expongas el puerto 8080 sin HTTPS: la contraseña viajaría sin cifrar.

## 6. Simulacro del kill switch (obligatorio antes de dinero real)

En horario de mercado:

```bash
docker compose run --rm bot tbot kill-drill
```

Compra 1 acción de SPY en paper, dispara el kill switch y verifica que se cierre todo.

## 7. Comandos útiles

| Qué | Cómo |
|---|---|
| Frenar todo ya | `/kill motivo` en Telegram, o `docker compose run --rm bot tbot kill --reason "motivo"` |
| Reactivar después de un kill | `docker compose run --rm bot tbot unkill` (pide escribir REACTIVAR) |
| Estado | `/status` en Telegram |
| Reporte del día | `docker compose run --rm bot tbot report` (también llega solo por Telegram a la noche) |
| Volver a la versión anterior de la estrategia | `docker compose run --rm bot tbot strategy-rollback && docker compose restart bot` |
| Actualizar el código | `git pull && docker compose up -d --build` |
| Volver a una versión anterior del código | `git checkout <commit> && docker compose up -d --build` |

## 8. Dinero real (solo cuando todo esté limpio)

1. Al menos 100 operaciones en paper.
2. `docker compose run --rm bot tbot golive-check` tiene que dar **PASA** en las cinco preguntas. Leé la sección
   "WHAT COULD BLOW UP THIS ACCOUNT?".
3. En Alpaca creá claves de la cuenta **real** solo con permiso de trading, **sin margen**. Ojo con la regla de
   day trading (PDT): con menos de US$25.000 en una cuenta de margen, más de 3 day trades en 5 días bloquea la cuenta.
4. Cambiá `mode = "live"` en `config/settings.toml`, poné las claves reales en `.env` y reiniciá.

El chequeo vale 24 horas y se invalida si cambiás cualquier límite de `config/risk.toml`: el bot se niega a operar en
real si no está vigente.
