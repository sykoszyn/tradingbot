"""Caché de velas en Parquet, por símbolo y mes. Los meses completos se bajan una sola vez."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd

Fetcher = Callable[[str, pd.Timestamp, pd.Timestamp], pd.DataFrame]


class BarCache:
    def __init__(self, root: Path, fetch: Fetcher):
        self.root = Path(root)
        self.fetch = fetch

    def _path(self, symbol: str, month: pd.Period) -> Path:
        return self.root / symbol / f"{month}.parquet"

    def get(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        now = pd.Timestamp.now(tz="UTC")
        parts = []
        for month in pd.period_range(start.tz_convert("UTC").tz_localize(None), end.tz_convert("UTC").tz_localize(None), freq="M"):
            m_start = month.start_time.tz_localize("UTC")
            m_end = (month + 1).start_time.tz_localize("UTC")
            path = self._path(symbol, month)
            complete = m_end < now - pd.Timedelta(days=1)
            if complete and path.exists():
                parts.append(pd.read_parquet(path))
                continue
            df = self.fetch(symbol, m_start, m_end)
            if complete:
                path.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(path, index=False)
            parts.append(df)
        out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if out.empty:
            return out
        out = out[(out["start"] >= start) & (out["start"] < end)]
        return out.drop_duplicates("start").sort_values("start").reset_index(drop=True)
