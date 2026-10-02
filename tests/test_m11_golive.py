import json

import pandas as pd
import pytest

from tbot.config import load_risk
from tbot.exec.alpaca_broker import AlpacaBroker
from tbot.golive.checklist import golive_ok, run_checklist, write_golive
from tbot.ledger.db import Ledger
from tbot.secrets import Secrets


def test_checklist_blocks_live_without_evidence(tmp_path):
    rep = run_checklist(Ledger(tmp_path / "l.sqlite3"), state_dir=tmp_path, strategy_dir=tmp_path, limits=load_risk())
    assert not rep.passed
    md = rep.markdown()
    assert "WHAT COULD BLOW UP THIS ACCOUNT?" in md and "NO PASA" in md
    for q in ("paper coincide con el backtest", "kill switch se disparó", "delegado a un modelo", "calibrada con fills propios", "régimen de mercado rompería"):
        assert q in md
    write_golive(rep, tmp_path / "nuevo" / "state", load_risk())  # instalación nueva: la carpeta todavía no existe
    assert not golive_ok(tmp_path / "nuevo" / "state", load_risk())


def test_checklist_passes_with_full_evidence(tmp_path):
    led = Ledger(tmp_path / "l.sqlite3")
    t = pd.Timestamp("2026-09-01 15:00", tz="UTC")
    for i in range(120):
        win = i % 5 != 0
        led.log_trade("SPY", t, t, 10, 100.0, 101.0 if win else 99.5, 10.0 if win else -5.0, 0.01 if win else -0.005, "take_profit" if win else "stop", 0.6)
    (tmp_path / "metrics.json").write_text(json.dumps({"hit_rate": 0.78, "avg_trade": 0.006, "por_régimen": {"trend": {"n_trades": 50, "hit_rate": 0.8, "avg_trade": 0.007}}, "por_año": {}}))
    (tmp_path / "kill_drill.json").write_text(json.dumps({"passed": True, "at": pd.Timestamp.now(tz="UTC").isoformat()}))
    (tmp_path / "calibration.json").write_text(json.dumps({"calibrated": True, "setup_quality": {"n": 300, "ece": 0.03, "brier": 0.2}}))
    rep = run_checklist(led, state_dir=tmp_path, strategy_dir=tmp_path, limits=load_risk())
    assert rep.passed, rep.markdown()
    write_golive(rep, tmp_path, load_risk())
    assert golive_ok(tmp_path, load_risk())


def test_golive_is_invalidated_when_risk_limits_change(tmp_path):
    (tmp_path / "golive.json").write_text(json.dumps({"passed": True, "at": pd.Timestamp.now(tz="UTC").isoformat(), "risk_fingerprint": "otra"}))
    assert not golive_ok(tmp_path, load_risk())


def test_live_broker_refuses_without_golive():
    with pytest.raises(PermissionError):
        AlpacaBroker(Secrets({"ALPACA_API_KEY": "k" * 20, "ALPACA_SECRET_KEY": "s" * 30}), paper=False, allow_live=False)


def test_cli_simulation_runs_end_to_end(tmp_path, monkeypatch):
    from tbot import cli, config
    from tbot.strategy.spec import StrategySpec

    sdir = tmp_path / "strategy"
    sdir.mkdir()
    StrategySpec(name="sim", template="trend", symbols=["SPY", "QQQ"]).save(sdir / "strategy.json")
    monkeypatch.setattr(cli, "STRATEGY_DIR", sdir)
    real = config.load_settings()
    fake = type(real)(**{**real.__dict__, "state_dir": tmp_path / "state", "ledger_path": tmp_path / "ledger.sqlite3", "scorer_path": tmp_path / "scorer.pkl", "data_dir": tmp_path / "data"})
    monkeypatch.setattr(cli, "load_settings", lambda: fake)
    assert cli.main(["run", "--sim", "--days", "3"]) == 0
    led = Ledger(tmp_path / "sim.sqlite3")
    assert len(led.decisions(limit=1000)) >= 3 * 26
