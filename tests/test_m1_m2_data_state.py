import numpy as np
import pandas as pd
import pytest

from tbot.data.bars import BarHistory, validate_bars
from tbot.data.cache import BarCache
from tbot.data.synthetic import make_bars
from tbot.state.engine import FEATURES, WARMUP_BARS, features_frame, snapshot


@pytest.fixture(scope="module")
def bars():
    return make_bars(days=60, seed=1)


def test_synthetic_bars_look_like_market_hours(bars):
    validate_bars(bars)
    et = bars["start"].dt.tz_convert("America/New_York")
    assert et.dt.weekday.max() <= 4
    assert (et.dt.hour * 60 + et.dt.minute).min() == 9 * 60 + 30
    assert (bars["high"] >= bars[["open", "close"]].max(axis=1)).all()


def test_history_never_returns_future_bars(bars):
    h = BarHistory("SPY", bars)
    t = bars["end"].iloc[100]
    view = h.upto(t)
    assert len(view) == 101
    assert (view["end"] <= t).all()
    # un instante antes del cierre de la vela 100, esa vela todavía no existe
    assert len(h.upto(t - pd.Timedelta(seconds=1))) == 100


def test_snapshot_ignores_future_bars(bars):
    k = 300
    f1 = features_frame(bars)
    mutated = bars.copy()
    rng = np.random.default_rng(7)
    for col in ("open", "high", "low", "close"):
        mutated.loc[k + 1 :, col] *= rng.uniform(0.5, 1.5, len(mutated) - k - 1)
    mutated.loc[k + 1 :, "volume"] *= 50
    f2 = features_frame(mutated)
    pd.testing.assert_frame_equal(f1.iloc[: k + 1], f2.iloc[: k + 1])


def test_live_snapshot_equals_backtest_row(bars):
    """El snapshot en vivo (con una ventana corta) es idéntico a la fila del backtest."""
    k = 400
    full = features_frame(bars)
    live = snapshot("SPY", bars.iloc[k - WARMUP_BARS - 10 : k + 1])
    assert live is not None
    np.testing.assert_allclose(live.vector(), full.loc[k, FEATURES].to_numpy(dtype=float), rtol=1e-9)
    assert live.ts == bars["end"].iloc[k]


def test_snapshot_needs_warmup(bars):
    assert snapshot("SPY", bars.iloc[: WARMUP_BARS - 1]) is None


def test_snapshot_is_compact_and_numeric(bars):
    s = snapshot("SPY", bars.iloc[:500])
    v = s.vector()
    assert v.dtype == float and len(v) == len(FEATURES) and np.isfinite(v).all()
    assert s.price > 0 and s.atr > 0


def test_cache_fetches_each_range_once(tmp_path, bars):
    calls = []

    def fetch(symbol, start, end):
        calls.append((symbol, start, end))
        return bars[(bars["start"] >= start) & (bars["start"] < end)]

    cache = BarCache(tmp_path, fetch)
    start, end = bars["start"].iloc[0], bars["start"].iloc[-1] + pd.Timedelta(minutes=15)
    a = cache.get("SPY", start, end)
    b = cache.get("SPY", start, end)
    assert len(a) == len(bars) and a.equals(b)
    n = len(calls)
    cache.get("SPY", start, end)
    assert len(calls) == n
