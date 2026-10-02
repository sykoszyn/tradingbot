import time

import numpy as np
import pandas as pd
import pytest

from tbot.backtest.costs import CostModel
from tbot.backtest.engine import Signal, run_backtest
from tbot.backtest.gate import GateThresholds, evaluate_gate, lookahead_check
from tbot.backtest.metrics import max_drawdown, sharpe, t_stat
from tbot.backtest.walkforward import walk_forward
from tbot.config import load_risk
from tbot.data.synthetic import make_bars
from tbot.scorer.calibration import brier, ece, reliability_curve
from tbot.scorer.labels import make_labels
from tbot.scorer.model import CalibratedScorer
from tbot.scorer.questions import QUESTIONS
from tbot.state.engine import FEATURES, features_frame
from tbot.strategy.spec import StrategySpec

ZERO = CostModel(spread_bps=0, slippage_atr_k=0, sec_fee_rate=0, taf_per_share=0)


def flat_bars(n=200, price=100.0):
    start = pd.Timestamp("2025-03-03 14:30", tz="UTC")
    rows = []
    for i in range(n):
        day = i // 26
        s = start + pd.Timedelta(days=day) + pd.Timedelta(minutes=15 * (i % 26))
        rows.append((s, s + pd.Timedelta(minutes=15), price, price + 0.05, price - 0.05, price, 1e5))
    return pd.DataFrame(rows, columns=["start", "end", "open", "high", "low", "close", "volume"])


def spec(**kw):
    base = dict(name="t", template="trend", symbols=["SPY"], horizon_bars=8, tp_atr=2.0, sl_atr=1.0, max_hold_bars=16, flatten_eod=False)
    base.update(kw)
    return StrategySpec(**base)


# ---------------- backtester ----------------

def test_take_profit_trade_has_exact_pnl():
    bars = flat_bars()
    k = 100
    bars.loc[k + 3, ["high", "close"]] = [103.0, 102.9]  # sube y toca el take profit (100 + 2 × 1)

    def decide(sym, i, feats, probs):
        return Signal(p_setup=0.9, b=2.0, reason="forzada") if i == k else None

    res = run_backtest({"SPY": bars}, spec(), decide=decide, atr_override=1.0, costs=ZERO, calibrated=True)
    assert len(res.trades) == 1
    t = res.trades[0]
    assert t.entry_price == 100.0 and t.exit_price == 102.0 and t.exit_reason == "take_profit"
    assert t.pnl == pytest.approx(t.qty * 2.0)


def test_stop_wins_when_both_hit_in_same_bar():
    bars = flat_bars()
    k = 100
    bars.loc[k + 2, ["high", "low"]] = [103.0, 98.5]

    def decide(sym, i, feats, probs):
        return Signal(p_setup=0.9, b=2.0, reason="f") if i == k else None

    res = run_backtest({"SPY": bars}, spec(), decide=decide, atr_override=1.0, costs=ZERO, calibrated=True)
    assert res.trades[0].exit_reason == "stop" and res.trades[0].exit_price == 99.0


def test_costs_reduce_pnl():
    bars = flat_bars()
    bars.loc[103, ["high", "close"]] = [103.0, 102.9]
    decide = lambda s, i, f, p: Signal(0.9, 2.0, "f") if i == 100 else None  # noqa: E731
    a = run_backtest({"SPY": bars}, spec(), decide=decide, atr_override=1.0, costs=ZERO, calibrated=True)
    b = run_backtest({"SPY": bars}, spec(), decide=decide, atr_override=1.0, costs=CostModel(), calibrated=True)
    assert b.trades[0].pnl < a.trades[0].pnl


def test_backtest_uses_the_real_risk_engine():
    bars = flat_bars()
    decide = lambda s, i, f, p: Signal(0.99, 5.0, "f") if i == 100 else None  # noqa: E731
    res = run_backtest({"SPY": bars}, spec(sl_atr=0.3, tp_atr=1.5), decide=decide, atr_override=1.0, costs=ZERO, calibrated=True)
    t = res.trades[0]
    assert t.qty * t.entry_price <= 100_000 * load_risk().max_position_pct + 1e-6


def test_uncalibrated_backtest_uses_minimum_size():
    bars = flat_bars()
    decide = lambda s, i, f, p: Signal(0.9, 2.0, "f") if i == 100 else None  # noqa: E731
    res = run_backtest({"SPY": bars}, spec(), decide=decide, atr_override=1.0, costs=ZERO, calibrated=False)
    assert res.trades[0].qty == int(100_000 * load_risk().min_position_pct / 100)


# ---------------- métricas ----------------

def test_metrics_on_known_series():
    eq = pd.Series([100, 110, 99, 120, 108], dtype=float)
    assert max_drawdown(eq) == pytest.approx(0.1)
    r = pd.Series([0.01, -0.005, 0.02, 0.0, 0.015])
    assert sharpe(r) == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(252))
    tr = np.array([0.01, 0.02, -0.01, 0.03])
    assert t_stat(tr) == pytest.approx(tr.mean() / (tr.std(ddof=1) / 2))


def test_gate_reports_each_failure():
    m = {"sharpe": 1.2, "max_drawdown": 0.2, "hit_rate": 0.5, "t_stat": 1.0, "years": 1.5, "n_trades": 10}
    g = evaluate_gate(m, GateThresholds())
    assert not g.passed and len(g.failures) == 6
    ok = {"sharpe": 1.8, "max_drawdown": 0.1, "hit_rate": 0.6, "t_stat": 2.5, "years": 2.2, "n_trades": 80}
    assert evaluate_gate(ok, GateThresholds()).passed


# ---------------- scorer ----------------

@pytest.fixture(scope="module")
def trained():
    bars = make_bars(days=300, seed=3)
    f = features_frame(bars)
    lab = make_labels(bars, f, horizon=8, tp_atr=1.5, sl_atr=1.0, max_hold=16)
    ok = f[FEATURES].notna().all(axis=1) & lab.notna().all(axis=1)
    X, Y = f.loc[ok, FEATURES].to_numpy(), lab.loc[ok]
    n = int(len(X) * 0.7)
    sc = CalibratedScorer(seed=0).fit(X[:n], Y.iloc[:n])
    return sc, X[n:], Y.iloc[n:]


def test_scorer_answers_every_question_with_valid_probabilities(trained):
    sc, X, _ = trained
    out = sc.score(X[0])
    assert set(out) == set(QUESTIONS)
    for q, classes in QUESTIONS.items():
        assert set(out[q]) == set(classes)
        assert sum(out[q].values()) == pytest.approx(1.0)
        assert all(0 <= v <= 1 for v in out[q].values())


def test_scorer_is_fast(trained):
    sc, X, _ = trained
    sc.score(X[0])
    t0 = time.perf_counter()
    for i in range(50):
        sc.score(X[i])
    assert (time.perf_counter() - t0) / 50 < 0.010


def test_fast_path_matches_sklearn_exactly(trained):
    sc, X, _ = trained
    batch = sc.score_batch(X[:200])
    for i in range(200):
        one = sc.score(X[i])
        for q, classes in QUESTIONS.items():
            np.testing.assert_allclose([one[q][c] for c in classes], batch[q][i], atol=1e-9)


def test_scorer_is_calibrated_when_the_signal_is_stable():
    """Con una probabilidad real conocida, lo que predice coincide con lo que pasa (fuera de muestra)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(9000, len(FEATURES)))
    p_true = 1 / (1 + np.exp(-2.0 * X[:, 0]))
    y = (rng.random(9000) < p_true).astype(float)
    labels = pd.DataFrame({q: y if len(c) == 2 else (y * 0) for q, c in QUESTIONS.items()})
    sc = CalibratedScorer(seed=0).fit(X[:6000], labels.iloc[:6000])
    p = sc.score_batch(X[6000:])["setup_quality"][:, 1]
    yt = y[6000:]
    assert ece(p, yt) < 0.04
    assert brier(p, yt) < brier(np.full_like(p, yt.mean()), yt) - 0.05  # informativo, no solo la tasa base
    assert np.corrcoef(p, p_true[6000:])[0, 1] > 0.9


def test_calibration_metrics_on_toy_data():
    p = np.array([0.9, 0.1, 0.8, 0.3])
    y = np.array([1, 0, 1, 1])
    assert brier(p, y) == pytest.approx(np.mean((p - y) ** 2))
    curve = reliability_curve(p, y, bins=2)
    assert curve["count"].sum() == 4
    assert ece(np.array([0.5] * 100), np.array([0, 1] * 50)) == pytest.approx(0.0)


# ---------------- walk-forward y filtro ----------------

@pytest.fixture(scope="module")
def long_bars():
    return {"SPY": make_bars(days=820, seed=11)}


def test_walk_forward_is_out_of_sample_and_purged(long_bars):
    wf = walk_forward(long_bars, spec(template="trend", flatten_eod=True), train_days=250, test_days=63)
    assert wf.metrics["years"] >= 2.0
    for fold in wf.folds:
        # ninguna etiqueta de entrenamiento mira más allá del comienzo del test
        assert fold.train_label_end < fold.test_start
    assert "sharpe" in wf.metrics and "by_year" in wf.metrics


def test_lookahead_cheat_is_detected(long_bars):
    # Un bug clásico: una feature que mira el precio futuro (shift negativo).
    def cheat_features(bars):
        f = features_frame(bars)
        f["peek"] = bars["close"].shift(-4) / bars["close"] - 1
        return f

    def cheat(sym, i, feats, probs):
        return Signal(0.9, 2.0, "trampa") if feats["peek"] > 0.004 else None

    def honest(sym, i, feats, probs):
        return Signal(0.9, 2.0, "honesta") if feats["ret_4"] > 0.004 else None

    assert not lookahead_check(long_bars, spec(), decide=cheat, feature_fn=cheat_features).passed
    assert lookahead_check(long_bars, spec(), decide=honest).passed


def test_markdown_renders_for_every_template():
    from tbot.strategy.spec import render_markdown

    for tpl in ("trend", "meanrev", "breakout"):
        md = render_markdown(spec(template=tpl))
        assert "Stop" in md and "Take profit" in md and "Invalidación" in md
