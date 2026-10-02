"""Reporte diario: operaciones, P&L, win rate, mayor pérdida, latencia y costo por decisión, calibración."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..ledger.db import Ledger
from ..scorer.live_calibration import calibration_status


@dataclass
class Report:
    data: dict
    markdown: str


def build_report(ledger: Ledger, day: dt.date, opus_cost_usd: float | None = None) -> Report:
    start = pd.Timestamp(day).tz_localize("America/New_York")
    end = start + pd.Timedelta(days=1)
    trades = ledger.trades_between(start, end)
    decisions = ledger.decisions_between(start, end)
    pnl = float(sum(t["pnl"] for t in trades))
    wins = [t for t in trades if t["pnl"] > 0]
    lat = [d["latency_ms"] for d in decisions if d["latency_ms"] is not None]
    opus = ledger.costs_between(start, end, "opus") if opus_cost_usd is None else opus_cost_usd
    cal = calibration_status(ledger)
    data = {
        "date": str(day),
        "trades": len(trades),
        "pnl": pnl,
        "win_rate": len(wins) / len(trades) if trades else 0.0,
        "largest_loss": float(min((t["pnl"] for t in trades), default=0.0)),
        "decisions": len(decisions),
        "signals": sum(1 for d in decisions if d["signal"]),
        "avg_latency_ms": float(np.mean(lat)) if lat else 0.0,
        "opus_cost_usd": opus,
        "cost_per_decision_usd": opus / len(decisions) if decisions else 0.0,
        "calibration": {q: {k: v for k, v in s.items() if k != "curve"} for q, s in cal.items() if isinstance(s, dict)},
        "calibrated": cal.get("calibrated", False),
    }
    sq = data["calibration"].get("setup_quality", {})
    md = "\n".join(
        [
            f"# Reporte {day}",
            "",
            f"- **Operaciones:** {data['trades']} · **P&L:** US${pnl:,.2f} · **Win rate:** {data['win_rate']:.0%} · **Mayor pérdida:** US${data['largest_loss']:,.2f}",
            f"- **Decisiones:** {data['decisions']} ({data['signals']} con señal) · **Latencia media de la capa rápida:** {data['avg_latency_ms']:.2f} ms",
            f"- **Costo:** Opus US${opus:.2f} · por decisión US${data['cost_per_decision_usd']:.4f} (la capa rápida corre local, sin costo por llamada)",
            f"- **Calibración (setup_quality):** n={sq.get('n', 0)}, Brier={sq.get('brier', '—')}, ECE={sq.get('ece', '—')} → "
            + ("✅ verificada: se usa ¼ Kelly" if data["calibrated"] else "⏳ sin verificar: tamaño mínimo fijo"),
        ]
    )
    return Report(data, md + "\n")
