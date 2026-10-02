import dataclasses
import logging

import pytest

from tbot.config import load_risk, load_settings
from tbot.secrets import RedactingFilter, Secrets


def test_risk_limits_loaded_from_file():
    r = load_risk()
    assert r.max_position_pct == 0.10
    assert r.max_gross_exposure_pct == 0.50
    assert r.daily_loss_limit_pct == 0.02
    assert r.max_drawdown_pct == 0.10
    assert r.manual_approval_usd == 1000
    assert r.max_trades_per_day == 10


def test_risk_limits_are_immutable():
    r = load_risk()
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.max_position_pct = 0.9  # type: ignore[misc]


def test_risk_fingerprint_changes_when_limits_change(tmp_path):
    a = load_risk()
    p = tmp_path / "risk.toml"
    p.write_text(open("config/risk.toml").read().replace("max_position_pct = 0.10", "max_position_pct = 0.20"))
    b = load_risk(p)
    assert a.fingerprint != b.fingerprint


def test_settings_paper_by_default():
    s = load_settings()
    assert s.mode == "paper"
    assert "SPY" in s.symbols


def test_secrets_never_printed(caplog):
    sec = Secrets({"ALPACA_API_KEY": "PKTESTKEY1234567890", "ALPACA_SECRET_KEY": "s3cr3tvalue-abcdef"})
    log = logging.getLogger("tbot.test")
    log.addFilter(RedactingFilter(sec))
    with caplog.at_level(logging.INFO, logger="tbot.test"):
        log.info("conectando con %s y %s", sec.get("ALPACA_API_KEY"), sec.get("ALPACA_SECRET_KEY"))
        log.info("clave en el mensaje: s3cr3tvalue-abcdef")
    text = caplog.text
    assert "PKTESTKEY1234567890" not in text and "s3cr3tvalue-abcdef" not in text
    assert "***" in text
    assert "PKTESTKEY" not in repr(sec)


def test_missing_secret_is_explicit():
    with pytest.raises(KeyError, match="ALPACA_API_KEY"):
        Secrets({}).get("ALPACA_API_KEY")
