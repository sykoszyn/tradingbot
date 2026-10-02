"""Chequeo final antes de dinero real. El modo real se niega a arrancar si algún punto no da bien."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ROOT, RiskLimits
from ..ledger.db import Ledger

MIN_PAPER_TRADES = 100
KILL_DRILL_MAX_AGE_DAYS = 7
GOLIVE_VALID_HOURS = 24


@dataclass
class Check:
    question: str
    passed: bool
    detail: str


@dataclass
class GoLiveReport:
    checks: list[Check]
    blowups: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def markdown(self) -> str:
        lines = ["# Chequeo final antes de dinero real", ""]
        for c in self.checks:
            lines.append(f"- {'✅' if c.passed else '❌'} **{c.question}** {c.detail}")
        lines += ["", "## WHAT COULD BLOW UP THIS ACCOUNT?", ""] + [f"- {b}" for b in self.blowups]
        lines += ["", f"**Resultado:** {'PASA: se puede activar el modo real a mano.' if self.passed else 'NO PASA: el modo real queda bloqueado.'}"]
        return "\n".join(lines) + "\n"


def _paper_vs_backtest(ledger: Ledger, strategy_dir: Path) -> Check:
    q = "¿El paper coincide con el backtest?"
    trades = ledger.trades(limit=10_000)
    mpath = strategy_dir / "metrics.json"
    if not mpath.exists():
        return Check(q, False, "no hay métricas de backtest de la estrategia aprobada")
    m = json.loads(mpath.read_text())
    if len(trades) < MIN_PAPER_TRADES:
        return Check(q, False, f"solo {len(trades)} operaciones en paper (mínimo {MIN_PAPER_TRADES})")
    rets = np.array([t["ret"] for t in trades])
    hit, avg = float((rets > 0).mean()), float(rets.mean())
    ok = abs(hit - m.get("hit_rate", 0)) <= 0.08 and avg > 0 and avg >= 0.5 * m.get("avg_trade", m.get("total_return", 0) / max(1, m.get("n_trades", 1)))
    return Check(q, ok, f"paper: hit {hit:.0%}, retorno medio {avg:.3%} · backtest: hit {m.get('hit_rate', 0):.0%}")


def _kill_drill(state_dir: Path) -> Check:
    q = "¿El kill switch se disparó en un test real?"
    p = state_dir / "kill_drill.json"
    if not p.exists():
        return Check(q, False, "nunca se corrió `tbot kill-drill`")
    d = json.loads(p.read_text())
    age = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(d["at"])).days
    ok = d.get("passed") and age <= KILL_DRILL_MAX_AGE_DAYS
    return Check(q, bool(ok), f"último simulacro hace {age} días: {'cerró todo' if d.get('passed') else 'FALLÓ'}")


def _no_model_owned_limits() -> Check:
    q = "¿Algún límite duro está delegado a un modelo?"
    src = ROOT / "src" / "tbot"
    problems = []
    for f in (src / "risk").glob("*.py"):
        if re.search(r"^\s*(from|import)\s+.*(brain|scorer|anthropic)", f.read_text(), re.M):
            problems.append(f"{f.name} importa un modelo")
    brain = "\n".join(f.read_text() for f in (src / "brain").glob("*.py"))
    if "risk.toml" in brain or "RiskLimits(" in brain:
        problems.append("el cerebro toca la config de riesgo")
    return Check(q, not problems, "no: los límites viven en config/risk.toml y solo los lee el código" if not problems else "; ".join(problems))


def _calibration(state_dir: Path) -> Check:
    q = "¿La confianza está calibrada con fills propios?"
    p = state_dir / "calibration.json"
    if not p.exists():
        return Check(q, False, "todavía no hay medición de calibración")
    c = json.loads(p.read_text())
    sq = c.get("setup_quality", {})
    return Check(q, bool(c.get("calibrated")), f"setup_quality: n={sq.get('n', 0)}, ECE={sq.get('ece', '—')}, Brier={sq.get('brier', '—')}")


def _regimes(strategy_dir: Path) -> tuple[Check, list[str]]:
    q = "¿Qué régimen de mercado rompería esto?"
    mpath = strategy_dir / "metrics.json"
    if not mpath.exists():
        return Check(q, False, "sin métricas por régimen"), []
    m = json.loads(mpath.read_text())
    weak = [f"{r} (hit {v['hit_rate']:.0%}, media {v['avg_trade']:.3%}, n={v['n_trades']})" for r, v in (m.get("por_régimen") or m.get("by_regime") or {}).items() if v.get("n_trades", 0) and (v["avg_trade"] < 0 or v["hit_rate"] < 0.5)]
    bad_years = [y for y, v in (m.get("por_año") or m.get("by_year") or {}).items() if v.get("return", 0) < 0]
    detail = ("pierde en: " + ", ".join(weak) if weak else "ningún régimen con expectativa negativa en el backtest") + (f"; años negativos: {', '.join(bad_years)}" if bad_years else "")
    return Check(q, True, detail), weak


def run_checklist(ledger: Ledger, *, state_dir: Path, strategy_dir: Path, limits: RiskLimits, spec=None) -> GoLiveReport:
    regime_check, weak = _regimes(strategy_dir)
    checks = [_paper_vs_backtest(ledger, strategy_dir), _kill_drill(state_dir), _no_model_owned_limits(), _calibration(state_dir), regime_check]
    blowups = [
        "**Gap de apertura o noticia fuera de horario:** si una posición queda abierta de noche, el stop no protege contra un gap; "
        + ("la estrategia cierra todo antes del final del día." if spec is None or spec.flatten_eod else "ESTA ESTRATEGIA MANTIENE POSICIONES DE NOCHE."),
        "**Movimiento violento (flash crash):** el stop es una orden de mercado al tocarse; en una caída rápida puede ejecutarse bastante peor.",
        f"**Una sola apuesta:** SPY y QQQ se mueven casi igual; con los dos abiertos es una única apuesta de hasta {2 * limits.max_position_pct:.0%} del capital.",
        "**Regla de day trading (PDT):** en una cuenta de margen con menos de US$25.000, más de 3 day trades en 5 días bloquea la cuenta. Usá cuenta cash o capital ≥ US$25.000.",
        "**Margen:** desactivá el margen en la cuenta real; los límites asumen que no hay apalancamiento.",
        "**Caída del bot o del servidor:** los stops y take profits quedan en Alpaca (bracket), pero las salidas por tiempo y el kill switch automático no corren con el bot caído. Hay reinicio automático y alerta.",
        "**Datos:** el feed gratuito (IEX) tiene menos volumen que el consolidado; si en vivo se usa otro feed que en el backtest, los resultados no son comparables.",
        "**Cambio de régimen:** " + ("la estrategia pierde en " + ", ".join(weak) + "." if weak else "el backtest no muestra un régimen perdedor, pero el futuro puede traer uno nuevo.") + " La invalidación automática la desactiva si el hit rate cae.",
        "**Sobreajuste nocturno:** cada cambio de Opus pasa el filtro fuera de muestra, pero muchas pruebas aumentan la chance de un falso positivo; revisá los cambios en strategy/history.",
        "**Claves:** usá claves solo con permiso de trading y nunca las pegues en un chat. Si se filtran, rotalas en Alpaca.",
    ]
    return GoLiveReport(checks, blowups)


def write_golive(report: GoLiveReport, state_dir: Path, limits: RiskLimits) -> None:
    Path(state_dir).mkdir(parents=True, exist_ok=True)
    (Path(state_dir) / "golive.json").write_text(json.dumps({"passed": report.passed, "at": pd.Timestamp.now(tz="UTC").isoformat(), "risk_fingerprint": limits.fingerprint}))


def golive_ok(state_dir: Path, limits: RiskLimits) -> bool:
    p = Path(state_dir) / "golive.json"
    if not p.exists():
        return False
    d = json.loads(p.read_text())
    fresh = pd.Timestamp.now(tz="UTC") - pd.Timestamp(d["at"]) < pd.Timedelta(hours=GOLIVE_VALID_HOURS)
    return bool(d.get("passed") and fresh and d.get("risk_fingerprint") == limits.fingerprint)
