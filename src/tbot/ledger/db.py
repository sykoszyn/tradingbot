"""Registro de todo: decisiones (con probabilidades y latencia), órdenes, fills, operaciones, eventos y capital."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
  id INTEGER PRIMARY KEY, ts TEXT, symbol TEXT, price REAL, features TEXT, probs TEXT, latency_ms REAL,
  signal INTEGER, action TEXT, reasons TEXT, client_id TEXT, strategy_version INTEGER, risk_fingerprint TEXT);
CREATE TABLE IF NOT EXISTS orders (
  client_id TEXT PRIMARY KEY, ts TEXT, symbol TEXT, side TEXT, qty INTEGER, ref_price REAL, stop REAL, tp REAL,
  status TEXT, reason TEXT, is_exit INTEGER, decision_id INTEGER, p_setup REAL);
CREATE TABLE IF NOT EXISTS fills (
  id INTEGER PRIMARY KEY, ts TEXT, client_id TEXT, symbol TEXT, side TEXT, qty INTEGER, price REAL, kind TEXT);
CREATE TABLE IF NOT EXISTS trades (
  id INTEGER PRIMARY KEY, symbol TEXT, entry_ts TEXT, exit_ts TEXT, qty INTEGER, entry_price REAL, exit_price REAL,
  pnl REAL, ret REAL, exit_reason TEXT, p_setup REAL, decision_id INTEGER);
CREATE TABLE IF NOT EXISTS outcomes (decision_id INTEGER, question TEXT, predicted TEXT, outcome INTEGER, PRIMARY KEY (decision_id, question));
CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, ts TEXT, kind TEXT, message TEXT);
CREATE TABLE IF NOT EXISTS equity (ts TEXT PRIMARY KEY, equity REAL, cash REAL);
CREATE TABLE IF NOT EXISTS costs (id INTEGER PRIMARY KEY, ts TEXT, kind TEXT, usd REAL);
"""


def _ts(t) -> str:
    return pd.Timestamp(t).tz_convert("UTC").isoformat() if pd.Timestamp(t).tzinfo else pd.Timestamp(t).tz_localize("UTC").isoformat()


class Ledger:
    def __init__(self, path: Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.executescript(SCHEMA)
            self._db.commit()

    def _exec(self, sql: str, args: tuple = ()) -> int:
        with self._lock:
            cur = self._db.execute(sql, args)
            self._db.commit()
            return int(cur.lastrowid or 0)

    def _rows(self, sql: str, args: tuple = ()) -> list[dict]:
        with self._lock:
            rows = [dict(r) for r in self._db.execute(sql, args).fetchall()]
        for r in rows:
            for k in ("features", "probs", "predicted"):
                if r.get(k):
                    r[k] = json.loads(r[k])
        return rows

    # ---- escritura ----
    def log_decision(self, ts, symbol, price, features, probs, latency_ms, signal, action, reasons, client_id, strategy_version, risk_fingerprint) -> int:
        return self._exec(
            "INSERT INTO decisions (ts,symbol,price,features,probs,latency_ms,signal,action,reasons,client_id,strategy_version,risk_fingerprint) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (_ts(ts), symbol, price, json.dumps(features), json.dumps(probs), latency_ms, int(bool(signal)), action, reasons, client_id, strategy_version, risk_fingerprint),
        )

    def log_order(self, ts, intent, status, decision_id=None, p_setup=None) -> None:
        self._exec(
            "INSERT OR REPLACE INTO orders (client_id,ts,symbol,side,qty,ref_price,stop,tp,status,reason,is_exit,decision_id,p_setup) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (intent.client_id, _ts(ts), intent.symbol, intent.side, intent.qty, intent.ref_price, intent.stop_price, intent.take_profit, status, intent.reason, int(intent.is_exit), decision_id, p_setup),
        )

    def has_fill(self, client_id: str, kind: str) -> bool:
        return bool(self._rows("SELECT 1 FROM fills WHERE client_id = ? AND kind = ? LIMIT 1", (client_id, kind)))

    def log_fill(self, fill) -> None:
        self._exec("INSERT INTO fills (ts,client_id,symbol,side,qty,price,kind) VALUES (?,?,?,?,?,?,?)", (_ts(fill.ts), fill.client_id, fill.symbol, fill.side, fill.qty, fill.price, fill.kind))

    def log_trade(self, symbol, entry_ts, exit_ts, qty, entry_price, exit_price, pnl, ret, exit_reason, p_setup, decision_id=None) -> int:
        return self._exec(
            "INSERT INTO trades (symbol,entry_ts,exit_ts,qty,entry_price,exit_price,pnl,ret,exit_reason,p_setup,decision_id) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (symbol, _ts(entry_ts), _ts(exit_ts), qty, entry_price, exit_price, pnl, ret, exit_reason, p_setup, decision_id),
        )

    def log_outcome(self, decision_id: int, question: str, predicted: dict, outcome: int) -> None:
        self._exec("INSERT OR REPLACE INTO outcomes (decision_id,question,predicted,outcome) VALUES (?,?,?,?)", (decision_id, question, json.dumps(predicted), int(outcome)))

    def log_event(self, kind: str, message: str, ts=None) -> int:
        return self._exec("INSERT INTO events (ts,kind,message) VALUES (?,?,?)", (_ts(ts or pd.Timestamp.now(tz="UTC")), kind, message))

    def log_equity(self, ts, equity: float, cash: float) -> None:
        self._exec("INSERT OR REPLACE INTO equity (ts,equity,cash) VALUES (?,?,?)", (_ts(ts), equity, cash))

    def log_cost(self, kind: str, usd: float, ts=None) -> None:
        self._exec("INSERT INTO costs (ts,kind,usd) VALUES (?,?,?)", (_ts(ts or pd.Timestamp.now(tz="UTC")), kind, usd))

    # ---- lectura ----
    def decisions(self, limit: int = 100, since_id: int = 0) -> list[dict]:
        return self._rows("SELECT * FROM decisions WHERE id > ? ORDER BY id DESC LIMIT ?", (since_id, limit))

    def decisions_between(self, start, end) -> list[dict]:
        return self._rows("SELECT * FROM decisions WHERE ts >= ? AND ts < ? ORDER BY id", (_ts(start), _ts(end)))

    def orders(self) -> list[dict]:
        return self._rows("SELECT * FROM orders ORDER BY ts")

    def order(self, client_id: str) -> dict | None:
        r = self._rows("SELECT * FROM orders WHERE client_id = ?", (client_id,))
        return r[0] if r else None

    def trades(self, limit: int = 1000) -> list[dict]:
        return self._rows("SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,))

    def trades_between(self, start, end) -> list[dict]:
        return self._rows("SELECT * FROM trades WHERE exit_ts >= ? AND exit_ts < ? ORDER BY id", (_ts(start), _ts(end)))

    def events(self, limit: int = 100, since_id: int = 0) -> list[dict]:
        return self._rows("SELECT * FROM events WHERE id > ? ORDER BY id DESC LIMIT ?", (since_id, limit))

    def equity_curve(self, limit: int = 5000) -> list[dict]:
        return list(reversed(self._rows("SELECT * FROM equity ORDER BY ts DESC LIMIT ?", (limit,))))

    def outcomes(self, question: str | None = None) -> list[dict]:
        if question:
            return self._rows("SELECT * FROM outcomes WHERE question = ?", (question,))
        return self._rows("SELECT * FROM outcomes")

    def costs_between(self, start, end, kind: str | None = None) -> float:
        q = "SELECT COALESCE(SUM(usd),0) AS s FROM costs WHERE ts >= ? AND ts < ?" + (" AND kind = ?" if kind else "")
        args = (_ts(start), _ts(end)) + ((kind,) if kind else ())
        return float(self._rows(q, args)[0]["s"])

    def entries_on(self, day_start, day_end) -> int:
        return int(self._rows("SELECT COUNT(*) AS n FROM orders WHERE is_exit = 0 AND status IN ('approved','filled') AND ts >= ? AND ts < ?", (_ts(day_start), _ts(day_end)))[0]["n"])
