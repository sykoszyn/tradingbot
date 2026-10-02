"""Loop en vivo, una llamada por cierre de vela. Los modelos aconsejan; este código decide y registra todo."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from ..config import RiskLimits
from ..exec.router import OrderRouter
from ..ledger.db import Ledger
from ..risk.sizing import position_shares
from ..risk.types import AccountState, OrderIntent
from ..state.engine import snapshot
from ..strategy.spec import StrategySpec
from ..strategy.templates import make_decider

log = logging.getLogger(__name__)
ET = "America/New_York"


class Notifier(Protocol):
    def notify(self, kind: str, text: str) -> None: ...
    def request_approval(self, pending_id: str, text: str) -> None: ...


class LogNotifier:
    def notify(self, kind: str, text: str) -> None:
        log.info("[%s] %s", kind, text)

    def request_approval(self, pending_id: str, text: str) -> None:
        log.warning("Aprobación pendiente %s (sin Telegram configurado, vence sola): %s", pending_id, text)


@dataclass
class RunnerState:
    peak_equity: float = 0.0
    day: str = ""
    day_start_equity: float = 0.0
    open_entries: dict[str, dict] = field(default_factory=dict)  # símbolo → entrada abierta
    last_reconcile: str = ""


def load_events(path: Path) -> list[tuple[pd.Timestamp, str]]:
    """config/events.csv: fecha,hora_et,nombre (por ejemplo 2026-10-28,14:00,FOMC)."""
    if not Path(path).exists():
        return []
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.lower().startswith("fecha"):
            continue
        d, t, name = (x.strip() for x in line.split(",", 2))
        out.append((pd.Timestamp(f"{d} {t}").tz_localize(ET), name))
    return out


class Runner:
    def __init__(
        self,
        *,
        spec: StrategySpec,
        scorer,
        router: OrderRouter,
        broker,
        ledger: Ledger,
        notifier: Notifier,
        fetch_recent: Callable[[str, pd.Timestamp], pd.DataFrame],
        limits: RiskLimits,
        state_dir: Path,
        decide=None,
        events: list[tuple[pd.Timestamp, str]] | None = None,
        observe: bool = False,
    ):
        self.spec, self.scorer, self.router, self.broker = spec, scorer, router, broker
        self.ledger, self.notifier, self.fetch_recent, self.limits = ledger, notifier, fetch_recent, limits
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.decide = decide or make_decider(spec)
        self.events = events or []
        self.observe = observe  # modo observación: registra predicciones y resultados, nunca manda órdenes
        self.tf = pd.Timedelta(minutes=spec.timeframe_minutes)
        self.state = self._load_state()
        self.now: pd.Timestamp | None = None
        self.calibrated: Callable[[], bool] = self._calibrated_from_file

    # ------------------------------------------------------------------ estado
    def _state_path(self) -> Path:
        return self.state_dir / "runner.json"

    def _load_state(self) -> RunnerState:
        try:
            return RunnerState(**json.loads(self._state_path().read_text()))
        except (OSError, ValueError, TypeError):
            return RunnerState()

    def _save_state(self) -> None:
        self._state_path().write_text(json.dumps(asdict(self.state)))

    def _calibrated_from_file(self) -> bool:
        try:
            return bool(json.loads((self.state_dir / "calibration.json").read_text()).get("calibrated"))
        except (OSError, ValueError):
            return False

    def strategy_disabled(self) -> str | None:
        p = self.state_dir / "STRATEGY_DISABLED"
        return p.read_text() if p.exists() else None

    def _account(self, now: pd.Timestamp) -> AccountState:
        equity, cash = self.broker.account()
        today = str(now.tz_convert(ET).date())
        if self.state.day != today:
            self.state.day, self.state.day_start_equity = today, equity
        self.state.peak_equity = max(self.state.peak_equity or equity, equity)
        d0 = pd.Timestamp(today).tz_localize(ET)
        trades_today = self.ledger.entries_on(d0, d0 + pd.Timedelta(days=1))
        return AccountState(equity, cash, self.broker.positions(), self.state.day_start_equity, self.state.peak_equity, trades_today, now, self.broker.market_open(now))

    # ------------------------------------------------------------------ fills
    def _process_fills(self) -> None:
        for f in self.broker.drain_fills():
            if self.ledger.has_fill(f.client_id, f.kind):
                continue  # ya registrado (por ejemplo, el broker lo vuelve a informar después de un reinicio)
            self.ledger.log_fill(f)
            if f.kind == "entry":
                o = self.ledger.order(f.client_id) or {}
                self.state.open_entries[f.symbol] = {"ts": pd.Timestamp(f.ts).isoformat(), "price": f.price, "qty": f.qty, "client_id": f.client_id, "p_setup": o.get("p_setup"), "decision_id": o.get("decision_id")}
                self.notifier.notify("fill", f"🟢 Compra {f.qty} {f.symbol} a US${f.price:,.2f}")
            else:
                e = self.state.open_entries.pop(f.symbol, None)
                if e:
                    pnl = (f.price - e["price"]) * f.qty
                    self.ledger.log_trade(f.symbol, e["ts"], f.ts, f.qty, e["price"], f.price, pnl, pnl / (e["price"] * f.qty), f.kind, e.get("p_setup"), e.get("decision_id"))
                    icon = "✅" if pnl > 0 else "🔻"
                    self.notifier.notify("fill", f"{icon} Venta {f.qty} {f.symbol} a US${f.price:,.2f} ({f.kind}) · P&L US${pnl:,.2f}")
                else:
                    self.notifier.notify("fill", f"Venta {f.qty} {f.symbol} a US${f.price:,.2f} ({f.kind})")

    # ------------------------------------------------------------------ controles
    def _reconcile(self, account: AccountState) -> None:
        broker_syms = {s for s, p in account.positions.items() if p.qty != 0}
        ours = set(self.state.open_entries)
        if broker_syms != ours:
            sig = f"{sorted(broker_syms)}|{sorted(ours)}"
            if sig != self.state.last_reconcile:
                msg = f"Diferencia con el broker: broker {sorted(broker_syms)} vs bot {sorted(ours)}. Manda el broker; revisá."
                self.ledger.log_event("reconcile", msg, self.now)
                self.notifier.notify("reconcile", "⚠️ " + msg)
                self.state.last_reconcile = sig
        else:
            self.state.last_reconcile = ""

    def _check_invalidation(self) -> None:
        if self.strategy_disabled():
            return
        inv = self.spec.invalidation
        trades = self.ledger.trades(limit=max(inv.rolling_trades, 200))
        reason = None
        recent = trades[: inv.rolling_trades]
        if len(recent) >= inv.rolling_trades:
            hit = np.mean([t["pnl"] > 0 for t in recent])
            if hit < inv.min_hit_rate:
                reason = f"hit rate {hit:.0%} < {inv.min_hit_rate:.0%} en las últimas {inv.rolling_trades} operaciones"
        if reason is None and trades:
            pnl = np.cumsum([t["pnl"] for t in reversed(trades)])
            base = self.state.peak_equity or 100_000
            dd = float(np.max(np.maximum.accumulate(pnl) - pnl)) / base
            if dd > inv.max_strategy_drawdown:
                reason = f"caída de la estrategia {dd:.1%} > {inv.max_strategy_drawdown:.0%}"
        if reason:
            (self.state_dir / "STRATEGY_DISABLED").write_text(reason)
            self.ledger.log_event("invalidation", reason, self.now)
            self.notifier.notify("invalidation", f"⛔ Estrategia desactivada: {reason}. No abre nada nuevo hasta que la revisión nocturna apruebe una versión nueva.")

    def _event_today(self, now: pd.Timestamp) -> tuple[pd.Timestamp, str] | None:
        if not self.spec.flatten_before_events:
            return None
        d = now.tz_convert(ET).date()
        return next(((t, n) for t, n in self.events if t.date() == d), None)

    def _expire_approvals(self, now: pd.Timestamp) -> None:
        ttl = pd.Timedelta(minutes=self.limits.approval_timeout_minutes)
        for pid, p in list(self.router.pending.items()):
            if now - p.created > ttl:
                self.router.reject(pid)
                self.notifier.notify("approval", f"⌛ Venció la aprobación {pid} ({p.intent.symbol}): no se ejecutó")

    def _exit(self, symbol: str, qty: int, price: float, reason: str, account: AccountState) -> None:
        intent = OrderIntent(symbol, "sell", qty, price, None, None, reason, is_exit=True)
        res = self.router.submit(intent, account)
        self.ledger.log_order(self.now, intent, res.status)

    def _manage_exits(self, account: AccountState, now: pd.Timestamp) -> None:
        et = now.tz_convert(ET)
        last_bar = et.hour * 60 + et.minute >= 16 * 60 - self.spec.timeframe_minutes
        ev = self._event_today(now)
        for s, p in account.positions.items():
            e = self.state.open_entries.get(s)
            held = int((now - pd.Timestamp(e["ts"])) / self.tf) if e else 0
            why = None
            if e and held >= self.spec.max_hold_bars:
                why = "salida por tiempo"
            elif self.spec.flatten_eod and last_bar:
                why = "cierre antes del fin del día"
            elif ev and now >= ev[0] - pd.Timedelta(minutes=15):
                why = f"cierre antes de {ev[1]}"
            if why and p.qty > 0:
                self._exit(s, p.qty, p.avg_price, why, account)

    # ------------------------------------------------------------------ principal
    def approve(self, pending_id: str):
        account = self._account(self.now or pd.Timestamp.now(tz="UTC"))
        res = self.router.approve(pending_id, account)
        self.ledger.log_order(account.now, res.intent, res.status)
        self.notifier.notify("approval", f"{'✅ Aprobada' if res.status == 'approved' else '❌ No se ejecutó'}: {res.intent.symbol} ({'; '.join(res.reasons)})")
        self._process_fills()
        self._save_state()
        return res

    def kill(self, reason: str) -> None:
        account = self._account(self.now or pd.Timestamp.now(tz="UTC"))
        self.router.kill(reason, account)
        self.ledger.log_event("kill", reason, account.now)
        self.notifier.notify("kill", f"🛑 KILL SWITCH: {reason}. Se cerró todo y el bot no opera hasta que lo reactives a mano (tbot unkill).")
        self._process_fills()
        self._save_state()

    def on_bar_close(self, now: pd.Timestamp) -> None:
        self.now = now
        self._process_fills()
        account = self._account(now)
        if not account.market_open:
            return
        for e in self.router.risk.on_equity(account):
            self.ledger.log_event(e.kind, e.message, now)
            if e.kind == "kill":
                self.kill(e.message)
                self.ledger.log_equity(now, *self.broker.account())
                return
            self.notifier.notify(e.kind, "⏸️ " + e.message)
        self._reconcile(account)
        self._check_invalidation()
        self._expire_approvals(now)
        self._manage_exits(account, now)
        self._process_fills()
        account = self._account(now)

        ev = self._event_today(now)
        for symbol in self.spec.symbols:
            bars = self.fetch_recent(symbol, now)
            snap = snapshot(symbol, bars)
            if snap is None:
                continue
            t0 = time.perf_counter()
            probs = self.scorer.score(snap.vector())
            latency = (time.perf_counter() - t0) * 1000
            feats = snap.as_dict()
            action, reasons, cid, did = "none", "", None, None
            sig = None
            if symbol in account.positions or symbol in self.state.open_entries:
                action = "holding"
            elif any(p.intent.symbol == symbol for p in self.router.pending.values()):
                action = "awaiting_approval"
            elif self.router.risk.killswitch.active:
                action = "killed"
            elif self.strategy_disabled():
                action = "strategy_disabled"
            elif ev:
                action = "blocked_event"
                reasons = f"día de {ev[1]}"
            else:
                sig = self.decide(symbol, len(bars) - 1, feats, probs)
            if sig is not None and self.observe:
                action, reasons = "observe_only", "modo observación: no opera"
            did = self.ledger.log_decision(now, symbol, snap.price, feats, probs, latency, sig is not None, action, reasons, None, self.spec.version, self.limits.fingerprint)
            if sig is None or self.observe:
                continue
            stop_dist = self.spec.sl_atr * snap.atr
            qty = position_shares(p=sig.p_setup, b=sig.b, stop_distance=stop_dist, price=snap.price, equity=account.equity, limits=self.limits, calibrated=self.calibrated(), p_min=0.0)
            if qty <= 0:
                self._update_decision(did, "size_zero", "Kelly sin ventaja")
                continue
            intent = OrderIntent(symbol, "buy", qty, snap.price, round(snap.price - stop_dist, 2), round(snap.price + self.spec.tp_atr * snap.atr, 2), sig.reason)
            res = self.router.submit(intent, account)
            self.ledger.log_order(now, intent, res.status, did, sig.p_setup)
            cid = intent.client_id
            if res.status == "approved":
                self._update_decision(did, "entry_submitted", "", cid)
            elif res.status == "needs_approval":
                self._update_decision(did, "needs_approval", "; ".join(res.reasons), cid)
                self.notifier.request_approval(res.pending_id, f"¿Comprar {qty} {symbol} a ~US${snap.price:,.2f} (US${intent.notional:,.0f})? Stop {intent.stop_price}, TP {intent.take_profit}. P(setup) {sig.p_setup:.0%}. Vence en {self.limits.approval_timeout_minutes} min.")
            else:
                self._update_decision(did, "rejected", "; ".join(res.reasons), cid)
            account = self._account(now)
        self._process_fills()
        eq, cash = self.broker.account()
        self.ledger.log_equity(now, eq, cash)
        self._save_state()

    def _update_decision(self, did: int, action: str, reasons: str, cid: str | None = None) -> None:
        self.ledger._exec("UPDATE decisions SET action = ?, reasons = ?, client_id = ? WHERE id = ?", (action, reasons, cid, did))
