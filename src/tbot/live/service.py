"""Servicio 24/7: corre el runner en cada cierre de vela, atiende Telegram y dispara las tareas nocturnas."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

import pandas as pd

log = logging.getLogger(__name__)
ET = "America/New_York"


def next_bar_close(now: pd.Timestamp, tf_minutes: int) -> pd.Timestamp:
    et = now.tz_convert(ET)
    open_ = et.normalize() + pd.Timedelta(hours=9, minutes=30)
    k = max(0, int((et - open_) / pd.Timedelta(minutes=tf_minutes)) + 1)
    return (open_ + pd.Timedelta(minutes=tf_minutes * k)).tz_convert("UTC")


def run_forever(*, runner, broker, tf_minutes: int, on_daily: Callable[[], None], nightly_hour_et: int, telegram=None, data_delay_s: int = 20, stop: threading.Event | None = None) -> None:
    stop = stop or threading.Event()
    if telegram is not None:

        def poll() -> None:
            while not stop.is_set():
                telegram.poll_once()

        threading.Thread(target=poll, daemon=True).start()
    last_daily = None
    while not stop.is_set():
        try:
            now = pd.Timestamp.now(tz="UTC")
            et = now.tz_convert(ET)
            if et.hour >= nightly_hour_et and last_daily != et.date():
                last_daily = et.date()
                on_daily()
            if not broker.market_open(now):
                stop.wait(60)
                continue
            target = next_bar_close(now, tf_minutes)
            wait = (target - now).total_seconds() + data_delay_s
            if wait > 0:
                stop.wait(min(wait, 60))
                if pd.Timestamp.now(tz="UTC") < target + pd.Timedelta(seconds=data_delay_s):
                    continue
            runner.on_bar_close(target)
        except Exception as e:  # nunca morir en silencio: avisar y seguir (el broker conserva stops y TPs)
            log.exception("Error en el loop")
            try:
                runner.notifier.notify("error", f"⚠️ Error en el loop: {type(e).__name__}: {str(e)[:300]}")
                runner.ledger.log_event("error", f"{type(e).__name__}: {str(e)[:500]}")
            except Exception:
                pass
            time.sleep(30)
