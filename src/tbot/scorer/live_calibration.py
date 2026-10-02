"""Calibración con decisiones propias: cuando madura el horizonte de una decisión, se registra lo que pasó
y se mide Brier, curva de confiabilidad y ECE por pregunta. Hasta que pase, el tamaño queda en el mínimo."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..ledger.db import Ledger
from ..state.engine import features_frame
from ..strategy.spec import StrategySpec
from .calibration import brier, ece, reliability_curve
from .labels import make_labels
from .questions import QUESTIONS

MIN_SAMPLES = 200
MAX_ECE = 0.05


def update_outcomes(ledger: Ledger, bars_by_symbol: dict[str, pd.DataFrame], spec: StrategySpec, start, end) -> int:
    """Calcula el resultado real de cada decisión entre start y end con las mismas etiquetas del entrenamiento."""
    n = 0
    done = {(o["decision_id"], o["question"]) for o in ledger.outcomes()}
    trades_by_decision = {t["decision_id"]: t for t in ledger.trades(limit=100_000) if t.get("decision_id")}
    for s, bars in bars_by_symbol.items():
        b = bars.reset_index(drop=True)
        f = features_frame(b)
        lab = make_labels(b, f, horizon=spec.horizon_bars, tp_atr=spec.tp_atr, sl_atr=spec.sl_atr, max_hold=spec.max_hold_bars)
        pos = {pd.Timestamp(t).isoformat(): i for i, t in enumerate(b["end"])}
        for d in ledger.decisions_between(start, end):
            if d["symbol"] != s or not d["probs"]:
                continue
            i = pos.get(pd.Timestamp(d["ts"]).isoformat())
            if i is None:
                continue
            for q in QUESTIONS:
                if (d["id"], q) in done or q not in d["probs"]:
                    continue
                y = lab[q].iloc[i]
                # para setup_quality, si la decisión se operó, manda el resultado del fill real
                t = trades_by_decision.get(d["id"])
                if q == "setup_quality" and t is not None:
                    y = 1.0 if t["exit_reason"] == "take_profit" else 0.0
                if pd.notna(y):
                    ledger.log_outcome(d["id"], q, d["probs"][q], int(y))
                    n += 1
    return n


def calibration_status(ledger: Ledger) -> dict:
    out: dict = {}
    for q, classes in QUESTIONS.items():
        rows = ledger.outcomes(q)
        if not rows:
            out[q] = {"n": 0}
            continue
        y = np.array([r["outcome"] for r in rows])
        P = np.array([[r["predicted"].get(c, 0.0) for c in classes] for r in rows])
        # one-vs-rest sobre la clase que usa la estrategia (o la positiva en las binarias)
        k = 1 if len(classes) == 2 else 0
        p, yk = P[:, k], (y == k).astype(float)
        out[q] = {
            "n": len(rows),
            "brier": round(brier(p, yk), 4),
            "ece": round(ece(p, yk), 4),
            "curve": reliability_curve(p, yk).dropna().round(3).to_dict("records"),
        }
    sq = out.get("setup_quality", {})
    out["calibrated"] = bool(sq.get("n", 0) >= MIN_SAMPLES and sq.get("ece", 1) < MAX_ECE)
    return out


def write_status(ledger: Ledger, state_dir: Path) -> dict:
    st = calibration_status(ledger)
    Path(state_dir).mkdir(parents=True, exist_ok=True)
    (Path(state_dir) / "calibration.json").write_text(json.dumps(st, ensure_ascii=False, indent=1))
    return st
