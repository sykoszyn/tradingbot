"""Velas históricas y recientes desde Alpaca (requiere ALPACA_API_KEY / ALPACA_SECRET_KEY)."""

from __future__ import annotations

import pandas as pd

from ..secrets import Secrets


def make_fetcher(secrets: Secrets, timeframe_minutes: int, feed: str = "iex"):
    from alpaca.data.enums import DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    client = StockHistoricalDataClient(secrets.get("ALPACA_API_KEY"), secrets.get("ALPACA_SECRET_KEY"))
    tf = TimeFrame(timeframe_minutes, TimeFrameUnit.Minute)

    def fetch(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=tf, start=start.to_pydatetime(), end=end.to_pydatetime(), feed=DataFeed(feed))
        raw = client.get_stock_bars(req).df
        if raw.empty:
            return pd.DataFrame(columns=["start", "end", "open", "high", "low", "close", "volume"])
        raw = raw.reset_index()
        raw = raw[raw["symbol"] == symbol] if "symbol" in raw else raw
        start_ts = pd.to_datetime(raw["timestamp"], utc=True)
        # solo horario regular (9:30–16:00 Nueva York)
        et = start_ts.dt.tz_convert("America/New_York")
        minutes = et.dt.hour * 60 + et.dt.minute
        mask = (minutes >= 570) & (minutes < 960)
        return pd.DataFrame(
            {
                "start": start_ts[mask].values,
                "end": (start_ts[mask] + pd.Timedelta(minutes=timeframe_minutes)).values,
                "open": raw["open"][mask].values,
                "high": raw["high"][mask].values,
                "low": raw["low"][mask].values,
                "close": raw["close"][mask].values,
                "volume": raw["volume"][mask].astype(float).values,
            }
        ).assign(start=lambda d: pd.to_datetime(d["start"], utc=True), end=lambda d: pd.to_datetime(d["end"], utc=True))

    return fetch
