"""Configuración: límites de riesgo (inmutables) y ajustes generales."""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RISK_PATH = ROOT / "config" / "risk.toml"
SETTINGS_PATH = ROOT / "config" / "settings.toml"


@dataclass(frozen=True)
class RiskLimits:
    max_position_pct: float
    max_gross_exposure_pct: float
    daily_loss_limit_pct: float
    max_drawdown_pct: float
    manual_approval_usd: float
    max_trades_per_day: int
    min_position_pct: float
    kelly_fraction: float
    approval_timeout_minutes: int
    fingerprint: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        for name in ("max_position_pct", "max_gross_exposure_pct", "daily_loss_limit_pct", "max_drawdown_pct", "min_position_pct", "kelly_fraction"):
            v = getattr(self, name)
            if not 0 < v <= 1:
                raise ValueError(f"{name} debe estar entre 0 y 1 (es {v})")
        if self.min_position_pct > self.max_position_pct:
            raise ValueError("min_position_pct no puede superar max_position_pct")


@dataclass(frozen=True)
class Settings:
    mode: str
    symbols: tuple[str, ...]
    timeframe_minutes: int
    data_dir: Path
    state_dir: Path
    ledger_path: Path
    strategy_path: Path
    scorer_path: Path
    nightly_hour_et: int
    opus_model: str
    opus_max_usd_per_night: float


def load_risk(path: Path | str = RISK_PATH) -> RiskLimits:
    raw = Path(path).read_bytes()
    data = tomllib.loads(raw.decode())
    return RiskLimits(**data, fingerprint=hashlib.sha256(raw).hexdigest()[:16])


def load_settings(path: Path | str = SETTINGS_PATH) -> Settings:
    d = tomllib.loads(Path(path).read_text())
    if d["mode"] not in ("paper", "live"):
        raise ValueError("mode debe ser 'paper' o 'live'")
    rel = lambda k: ROOT / d[k]  # noqa: E731
    return Settings(
        mode=d["mode"],
        symbols=tuple(d["symbols"]),
        timeframe_minutes=int(d["timeframe_minutes"]),
        data_dir=rel("data_dir"),
        state_dir=rel("state_dir"),
        ledger_path=rel("ledger_path"),
        strategy_path=rel("strategy_path"),
        scorer_path=rel("scorer_path"),
        nightly_hour_et=int(d["nightly_hour_et"]),
        opus_model=d["opus_model"],
        opus_max_usd_per_night=float(d["opus_max_usd_per_night"]),
    )
