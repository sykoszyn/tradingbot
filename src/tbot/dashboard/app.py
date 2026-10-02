"""Dashboard en vivo: cada señal con su probabilidad, confianza, acción y resultado. Protegido con contraseña."""

from __future__ import annotations

import asyncio
import json
import secrets as pysecrets
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from ..ledger.db import Ledger

HTML = (Path(__file__).parent / "index.html").read_text()


def _confidence(probs: dict) -> float | None:
    try:
        return round((probs["setup_quality"]["success"] + probs["direction"]["up"] + probs["pressure_real"]["yes"] + max(probs["regime"]["trend"], probs["regime"]["range"])) / 4, 4)
    except (KeyError, TypeError):
        return None


def create_app(ledger: Ledger, *, state_dir: Path, password: str, strategy_md: Path | None = None) -> FastAPI:
    if not password or len(password) < 10:
        raise ValueError("Configurá DASHBOARD_PASSWORD (mínimo 10 caracteres) antes de exponer el dashboard")
    app = FastAPI(title="tbot", docs_url=None, redoc_url=None)
    security = HTTPBasic()

    def auth(c: HTTPBasicCredentials = Depends(security)) -> None:  # noqa: B008
        ok_user = pysecrets.compare_digest(c.username.encode(), b"admin")
        ok_pass = pysecrets.compare_digest(c.password.encode(), password.encode())
        if not (ok_user and ok_pass):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, headers={"WWW-Authenticate": "Basic"})

    def decision_rows(limit: int = 200, since_id: int = 0) -> list[dict]:
        rows = ledger.decisions(limit=limit, since_id=since_id)
        by_dec = {t["decision_id"]: t for t in ledger.trades(limit=2000) if t.get("decision_id")}
        out = []
        for d in rows:
            p = d.get("probs") or {}
            t = by_dec.get(d["id"])
            out.append(
                {
                    "id": d["id"], "ts": d["ts"], "symbol": d["symbol"], "price": d["price"], "action": d["action"], "reasons": d["reasons"],
                    "signal": bool(d["signal"]), "probability": (p.get("setup_quality") or {}).get("success"), "confidence": _confidence(p),
                    "latency_ms": d["latency_ms"], "result": None if t is None else {"pnl": t["pnl"], "exit": t["exit_reason"]},
                }
            )
        return out

    def summary() -> dict:
        eq = ledger.equity_curve(limit=2000)
        kill = Path(state_dir) / "KILL"
        disabled = Path(state_dir) / "STRATEGY_DISABLED"
        cal = Path(state_dir) / "calibration.json"
        trades = ledger.trades(limit=500)
        return {
            "equity": eq[-1]["equity"] if eq else None,
            "equity_curve": [[e["ts"], e["equity"]] for e in eq[-600:]],
            "kill": json.loads(kill.read_text()) if kill.exists() else None,
            "strategy_disabled": disabled.read_text() if disabled.exists() else None,
            "calibration": json.loads(cal.read_text()) if cal.exists() else None,
            "trades": len(trades),
            "pnl_total": sum(t["pnl"] for t in trades),
            "win_rate": (sum(t["pnl"] > 0 for t in trades) / len(trades)) if trades else None,
            "events": ledger.events(limit=20),
            "strategy_md": strategy_md.read_text() if strategy_md and strategy_md.exists() else None,
        }

    @app.get("/", response_class=HTMLResponse, dependencies=[Depends(auth)])
    def index() -> str:
        return HTML

    @app.get("/api/summary", dependencies=[Depends(auth)])
    def api_summary() -> dict:
        return summary()

    @app.get("/api/decisions", dependencies=[Depends(auth)])
    def api_decisions(limit: int = 200) -> list[dict]:
        return decision_rows(min(limit, 1000))

    @app.get("/api/stream", dependencies=[Depends(auth)])
    async def stream() -> StreamingResponse:
        async def gen():
            last = (ledger.decisions(limit=1) or [{"id": 0}])[0]["id"]
            last_ev = (ledger.events(limit=1) or [{"id": 0}])[0]["id"]
            while True:
                new = decision_rows(limit=200, since_id=last)
                evs = ledger.events(limit=50, since_id=last_ev)
                if new or evs:
                    last = max([last] + [d["id"] for d in new])
                    last_ev = max([last_ev] + [e["id"] for e in evs])
                    yield f"data: {json.dumps({'decisions': list(reversed(new)), 'events': list(reversed(evs))})}\n\n"
                else:
                    yield ": ping\n\n"
                await asyncio.sleep(2)

        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-store"})

    return app
