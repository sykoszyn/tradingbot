"""Velas sintéticas con cambios de régimen, para tests y para probar el sistema sin conexión."""

from __future__ import annotations

import numpy as np
import pandas as pd

BARS_PER_DAY = 26  # 9:30 a 16:00 en velas de 15 minutos


def make_bars(days: int = 250, seed: int = 0, start: str = "2023-01-03", price: float = 400.0, tf_minutes: int = 15) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=days)
    per_day = int(390 / tf_minutes)
    # regímenes diarios con persistencia: 0 tendencia alcista, 1 bajista, 2 rango, 3 alta volatilidad
    regimes = np.empty(days, dtype=int)
    r = 2
    for i in range(days):
        if rng.random() < 0.08:
            r = rng.choice(4, p=[0.35, 0.2, 0.35, 0.1])
        regimes[i] = r
    drift = {0: 0.00025, 1: -0.0003, 2: 0.0, 3: 0.0}
    vol = {0: 0.0016, 1: 0.002, 2: 0.0013, 3: 0.0045}
    rows = []
    p = price
    anchor = price
    for d, day in enumerate(dates):
        reg = regimes[d]
        open_et = pd.Timestamp(day.date()).tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=30)
        if reg == 2 and d % 5 == 0:
            anchor = p
        for b in range(per_day):
            start_ts = (open_et + pd.Timedelta(minutes=tf_minutes * b)).tz_convert("UTC")
            mr = -0.08 * np.log(p / anchor) if reg == 2 else 0.0
            ret = drift[reg] + mr + vol[reg] * rng.standard_t(5) / np.sqrt(5 / 3)
            o = p
            c = p * np.exp(ret)
            wick = abs(rng.normal(0, vol[reg] * 0.6))
            h = max(o, c) * (1 + wick)
            low = min(o, c) * (1 - abs(rng.normal(0, vol[reg] * 0.6)))
            u = 1.6 - np.sin(np.pi * b / max(1, per_day - 1))  # volumen en forma de U
            v = float(rng.lognormal(np.log(2e5 * u * (1 + 40 * abs(ret))), 0.3))
            rows.append((start_ts, start_ts + pd.Timedelta(minutes=tf_minutes), o, h, low, c, v))
            p = c
    df = pd.DataFrame(rows, columns=["start", "end", "open", "high", "low", "close", "volume"])
    df.attrs["regimes"] = regimes.tolist()
    return df
