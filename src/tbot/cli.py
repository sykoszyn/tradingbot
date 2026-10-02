"""Línea de comandos: `tbot <comando>`. Ver `tbot --help`."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from .config import ROOT, load_risk, load_settings
from .secrets import Secrets, setup_logging

log = logging.getLogger("tbot")
STRATEGY_DIR = ROOT / "strategy"


# ---------------------------------------------------------------------------- utilidades

def _load_bars(settings, secrets, *, years: float, synthetic: bool) -> dict[str, pd.DataFrame]:
    if synthetic:
        from .data.synthetic import make_bars

        days = int(252 * years)
        return {s: make_bars(days=days, seed=i + 1, price=400 + 50 * i) for i, s in enumerate(settings.symbols)}
    from .data.alpaca import make_fetcher
    from .data.cache import BarCache

    cache = BarCache(settings.data_dir, make_fetcher(secrets, settings.timeframe_minutes))
    end = pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(days=int(365.25 * years))
    return {s: cache.get(s, start, end) for s in settings.symbols}


def _load_spec(observe: bool = False):
    from .strategy.spec import StrategySpec

    p = STRATEGY_DIR / "strategy.json"
    if p.exists():
        return StrategySpec.load(p), False
    c = STRATEGY_DIR / "candidate.json"
    if observe and c.exists():
        return StrategySpec.load(c), True
    return None, False


def _notifier(secrets, runner_ref: dict):
    from .alerts.telegram import TelegramBot, TelegramNotifier
    from .live.runner import LogNotifier

    token, chat = secrets.optional("TELEGRAM_BOT_TOKEN"), secrets.optional("TELEGRAM_CHAT_ID")
    if not (token and chat):
        return LogNotifier(), None
    import httpx

    bot = TelegramBot(
        token=token,
        chat_id=chat,
        on_approve=lambda pid: runner_ref["r"].approve(pid),
        on_reject=lambda pid: runner_ref["r"].router.reject(pid),
        on_kill=lambda reason: runner_ref["r"].kill(reason),
        on_status=lambda: _status_text(runner_ref.get("r")),
        http=httpx.Client(),
    )
    return TelegramNotifier(bot), bot


def _status_text(runner) -> str:
    if runner is None:
        return "Iniciando…"
    eq, cash = runner.broker.account()
    kill = runner.router.risk.killswitch
    return (
        f"Capital US${eq:,.2f} · efectivo US${cash:,.2f}\nPosiciones: {', '.join(f'{p.qty} {s}' for s, p in runner.broker.positions().items()) or 'ninguna'}\n"
        f"Estrategia: {runner.spec.name} v{runner.spec.version}{' (observación)' if runner.observe else ''}\n"
        f"Kill switch: {'ACTIVO: ' + str(kill.reason()) if kill.active else 'no'}\nEstrategia desactivada: {runner.strategy_disabled() or 'no'}"
    )


def _scorer(settings, spec, bars, retrain: bool = False):
    from .backtest.walkforward import walk_forward
    from .scorer.model import CalibratedScorer

    if settings.scorer_path.exists() and not retrain:
        return CalibratedScorer.load(settings.scorer_path)
    wf = walk_forward(bars, spec, fit_final=True)
    wf.final_scorer.save(settings.scorer_path)
    return wf.final_scorer


# ---------------------------------------------------------------------------- comandos

def cmd_backtest(a, settings, secrets) -> int:
    from .backtest.gate import GateThresholds, evaluate_gate, lookahead_check
    from .backtest.walkforward import walk_forward
    from .strategy.spec import StrategySpec, render_markdown

    bars = _load_bars(settings, secrets, years=a.years, synthetic=a.synthetic)
    spec = StrategySpec.load(Path(a.spec)) if a.spec else StrategySpec(name=a.template, template=a.template, symbols=list(settings.symbols))
    wf = walk_forward(bars, spec)
    g = evaluate_gate(wf.metrics, GateThresholds.load())
    la = lookahead_check(bars, spec)
    print(render_markdown(spec, wf.metrics, {"passed": g.passed and la.passed, "failures": g.failures + la.failures}))
    return 0


def cmd_research(a, settings, secrets) -> int:
    from .backtest.gate import GateThresholds
    from .brain.opus import OpusClient
    from .brain.research import candidate_to_spec, research

    bars = _load_bars(settings, secrets, years=a.years, synthetic=a.synthetic)
    client = OpusClient(secrets.get("ANTHROPIC_API_KEY"), model=settings.opus_model)
    res = research(bars, client=client, gate=GateThresholds.load(), out_dir=STRATEGY_DIR, rounds=a.rounds, max_usd=settings.opus_max_usd_per_night)
    from .ledger.db import Ledger

    Ledger(settings.ledger_path).log_cost("opus", res.spent_usd)
    print(f"Gastado en Opus: US${res.spent_usd:.2f} · fin: {res.stopped_reason}")
    if res.winner:
        print(f"✅ Ganadora: {res.winner.name}. Ver strategy/strategy.md")
        _scorer(settings, res.winner, bars, retrain=True)
        return 0
    scored = [x for x in res.log if x.get("métricas")]
    if scored:
        best = max(scored, key=lambda x: x["métricas"]["sharpe"])
        candidate_to_spec(best["candidata"], list(settings.symbols)).save(STRATEGY_DIR / "candidate.json")
        print(f"❌ Ninguna pasó el filtro. La mejor ({best['name']}) quedó en strategy/candidate.json para el modo observación (tbot run --observe).")
    return 1


def cmd_run(a, settings, secrets) -> int:
    import time

    from .exec.router import OrderRouter
    from .golive.checklist import golive_ok
    from .ledger.db import Ledger
    from .live.runner import Runner, load_events
    from .risk.killswitch import KillSwitch
    from .risk.limits import RiskEngine

    limits = load_risk()
    spec, observe_only = _load_spec(observe=a.observe or a.auto)
    while spec is None and a.auto and not a.sim:
        log.warning("No hay estrategia aprobada ni candidata: corré `tbot research`. Reintento en 1 hora.")
        (settings.state_dir).mkdir(parents=True, exist_ok=True)
        (settings.state_dir / "heartbeat").write_text(pd.Timestamp.now(tz="UTC").isoformat())
        time.sleep(3600)
        spec, observe_only = _load_spec(observe=True)
    if spec is None:
        print("No hay estrategia aprobada (strategy/strategy.json). Corré `tbot research`, o `tbot run --observe` con una candidata.")
        return 1
    # --observe fuerza la observación; --auto observa solo si no hay estrategia aprobada
    observe = a.observe or observe_only
    live = settings.mode == "live" and not a.sim
    ok_live = golive_ok(settings.state_dir, limits)
    if live and not ok_live:
        print("Modo real bloqueado: `tbot golive-check` no pasó en las últimas 24 h (o cambiaron los límites).")
        return 2
    ledger = Ledger(settings.ledger_path if not a.sim else settings.ledger_path.with_name("sim.sqlite3"))
    risk = RiskEngine(limits, KillSwitch(settings.state_dir), universe=set(spec.symbols), mode="live" if live else "paper", golive_ok=ok_live)
    ref: dict = {}
    notifier, bot = _notifier(secrets, ref)

    if a.sim:
        from .data.synthetic import make_bars
        from .exec.paper_sim import PaperSimBroker
        from .scorer.model import CalibratedScorer  # noqa: F401

        bars = {s: make_bars(days=a.days + 420, seed=i + 7, price=400 + 50 * i) for i, s in enumerate(spec.symbols)}
        train = {s: b.iloc[: -a.days * 26] for s, b in bars.items()}
        scorer = _scorer(settings, spec, train, retrain=True)
        broker = PaperSimBroker()
        clock = {"now": None}
        runner = Runner(spec=spec, scorer=scorer, router=OrderRouter(broker, risk), broker=broker, ledger=ledger, notifier=notifier,
                        fetch_recent=lambda s, now: bars[s][bars[s]["end"] <= now].tail(200), limits=limits, state_dir=settings.state_dir / "sim", observe=observe)
        ref["r"] = runner
        ends = sorted(set().union(*[set(b["end"].iloc[-a.days * 26 :]) for b in bars.values()]))
        for t in ends:
            clock["now"] = t
            for s, b in bars.items():
                row = b[b["end"] == t]
                if len(row):
                    broker.on_bar(s, row.iloc[0])
            runner.on_bar_close(t)
        print(_status_text(runner))
        print(f"Simulación lista: {a.days} días. Mirá el dashboard con `tbot dashboard --sim`.")
        return 0

    from .data.alpaca import make_fetcher
    from .exec.alpaca_broker import AlpacaBroker
    from .live.service import run_forever

    fetch = make_fetcher(secrets, spec.timeframe_minutes)
    bars = _load_bars(settings, secrets, years=2.5, synthetic=False)
    scorer = _scorer(settings, spec, bars)
    broker = AlpacaBroker(secrets, paper=not live, allow_live=ok_live)
    runner = Runner(spec=spec, scorer=scorer, router=OrderRouter(broker, risk), broker=broker, ledger=ledger, notifier=notifier,
                    fetch_recent=lambda s, now: fetch(s, now - pd.Timedelta(days=7), now + pd.Timedelta(seconds=1)).pipe(lambda d: d[d["end"] <= now]),
                    limits=limits, state_dir=settings.state_dir, events=load_events(ROOT / "config" / "events.csv"), observe=observe)
    ref["r"] = runner
    notifier.notify("start", f"▶️ Bot iniciado en {'REAL' if live else 'paper'}{' (observación)' if observe else ''} · {spec.name} v{spec.version}")
    run_forever(runner=runner, broker=broker, tf_minutes=spec.timeframe_minutes, on_daily=lambda: _nightly(settings, secrets, runner, notifier), nightly_hour_et=settings.nightly_hour_et, telegram=bot, heartbeat=settings.state_dir / "heartbeat")
    return 0


def _nightly(settings, secrets, runner, notifier) -> None:
    from .backtest.gate import GateThresholds
    from .brain.nightly import nightly_review
    from .brain.opus import OpusClient
    from .report.daily import build_report
    from .scorer.live_calibration import update_outcomes, write_status

    ledger = runner.ledger
    today = pd.Timestamp.now(tz="America/New_York").normalize()
    bars = _load_bars(settings, secrets, years=2.5, synthetic=False)
    update_outcomes(ledger, bars, runner.spec, today - pd.Timedelta(days=10), today + pd.Timedelta(days=1))
    write_status(ledger, settings.state_dir)
    rep = build_report(ledger, today.date())
    out = settings.state_dir.parent / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{today.date()}.md").write_text(rep.markdown)
    notifier.notify("report", rep.markdown)
    key = secrets.optional("ANTHROPIC_API_KEY")
    if not key or not (STRATEGY_DIR / "strategy.json").exists():
        return
    day = {
        "trades": ledger.trades_between(today, today + pd.Timedelta(days=1)),
        "decisiones_con_señal": [d for d in ledger.decisions_between(today, today + pd.Timedelta(days=1)) if d["signal"]],
        "eventos": ledger.events(limit=50),
        "reporte": rep.data,
    }
    res = nightly_review(bars, day_data=day, client=OpusClient(key, model=settings.opus_model), gate=GateThresholds.load(), strategy_dir=STRATEGY_DIR, max_usd=settings.opus_max_usd_per_night)
    ledger.log_cost("opus", res.spent_usd)
    if res.shipped:
        from .strategy.spec import StrategySpec

        runner.spec = StrategySpec.load(STRATEGY_DIR / "strategy.json")
        runner.scorer = _scorer(settings, runner.spec, bars, retrain=True)
        (settings.state_dir / "STRATEGY_DISABLED").unlink(missing_ok=True)
        notifier.notify("nightly", f"🧠 Nueva versión v{runner.spec.version} aplicada (pasó el filtro). Regla nueva: {res.new_rule}")
    else:
        notifier.notify("nightly", f"🧠 Revisión: {res.analysis[:800]}\nRegla nueva: {res.new_rule}\nSin cambios en la estrategia{': ' + '; '.join(res.failures[:3]) if res.failures else ''}.")


def cmd_nightly(a, settings, secrets) -> int:
    print("La tarea nocturna corre sola dentro de `tbot run` a la hora configurada (nightly_hour_et).")
    return 0


def cmd_dashboard(a, settings, secrets) -> int:
    import uvicorn

    from .dashboard.app import create_app
    from .ledger.db import Ledger

    path = settings.ledger_path.with_name("sim.sqlite3") if a.sim else settings.ledger_path
    state = settings.state_dir / "sim" if a.sim else settings.state_dir
    app = create_app(Ledger(path), state_dir=state, password=secrets.get("DASHBOARD_PASSWORD"), strategy_md=STRATEGY_DIR / "strategy.md")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
    return 0


def cmd_kill(a, settings, secrets) -> int:
    from .risk.killswitch import KillSwitch

    KillSwitch(settings.state_dir).fire(a.reason)
    print("Kill switch ACTIVADO. Si el bot está corriendo, cierra todo en la próxima vela. Para cerrar ya, usá /kill en Telegram.")
    return 0


def cmd_unkill(a, settings, secrets) -> int:
    from .risk.killswitch import CONFIRM_WORD, KillSwitch

    ks = KillSwitch(settings.state_dir)
    if not ks.active:
        print("El kill switch no está activo.")
        return 0
    print(f"Motivo: {ks.reason()}\nPara reactivar el trading escribí {CONFIRM_WORD}:")
    ks.reset(input().strip())
    print("Reactivado.")
    return 0


def cmd_kill_drill(a, settings, secrets) -> int:
    """Simulacro real en paper: compra 1 acción, dispara el kill switch y verifica que se cierre todo."""
    import time

    from .exec.alpaca_broker import AlpacaBroker
    from .exec.router import OrderRouter
    from .ledger.db import Ledger
    from .live.runner import LogNotifier, Runner
    from .risk.killswitch import CONFIRM_WORD, KillSwitch
    from .risk.limits import RiskEngine
    from .risk.types import OrderIntent

    if settings.mode != "paper":
        print("El simulacro se hace solo en paper.")
        return 2
    limits = load_risk()
    broker = AlpacaBroker(secrets, paper=True)
    if not broker.market_open(pd.Timestamp.now(tz="UTC")):
        print("El mercado está cerrado: corré el simulacro en horario de mercado.")
        return 2
    risk = RiskEngine(limits, KillSwitch(settings.state_dir), universe={"SPY"}, mode="paper")
    router = OrderRouter(broker, risk)
    spec, _ = _load_spec(observe=True)
    from .strategy.spec import StrategySpec

    runner = Runner(spec=spec or StrategySpec(name="drill", template="trend", symbols=["SPY"]), scorer=None, router=router, broker=broker, ledger=Ledger(settings.ledger_path), notifier=LogNotifier(), fetch_recent=lambda s, n: pd.DataFrame(), limits=limits, state_dir=settings.state_dir)
    now = pd.Timestamp.now(tz="UTC")
    acct = runner._account(now)
    res = router.submit(OrderIntent("SPY", "buy", 1, 600.0, 1.0, 10_000.0, "simulacro de kill switch"), acct)
    print("Compra de prueba:", res.status, res.reasons)
    time.sleep(5)
    runner.kill("simulacro")
    time.sleep(8)
    ok = "SPY" not in broker.positions()
    (settings.state_dir / "kill_drill.json").write_text(json.dumps({"passed": ok, "at": pd.Timestamp.now(tz="UTC").isoformat()}))
    KillSwitch(settings.state_dir).reset(CONFIRM_WORD)
    print("✅ El kill switch cerró todo." if ok else "❌ Quedó una posición abierta: revisá en Alpaca.")
    return 0 if ok else 1


def cmd_golive(a, settings, secrets) -> int:
    from .golive.checklist import run_checklist, write_golive
    from .ledger.db import Ledger

    limits = load_risk()
    spec, _ = _load_spec()
    rep = run_checklist(Ledger(settings.ledger_path), state_dir=settings.state_dir, strategy_dir=STRATEGY_DIR, limits=limits, spec=spec)
    write_golive(rep, settings.state_dir, limits)
    print(rep.markdown())
    return 0 if rep.passed else 1


def cmd_report(a, settings, secrets) -> int:
    from .ledger.db import Ledger
    from .report.daily import build_report

    day = pd.Timestamp(a.date).date() if a.date else pd.Timestamp.now(tz="America/New_York").date()
    print(build_report(Ledger(settings.ledger_path), day).markdown)
    return 0


def cmd_status(a, settings, secrets) -> int:
    from .risk.killswitch import KillSwitch

    ks = KillSwitch(settings.state_dir)
    spec, _ = _load_spec(observe=True)
    print(f"Modo: {settings.mode} · Estrategia: {spec.name + ' v' + str(spec.version) if spec else 'ninguna'} · Kill switch: {'ACTIVO (' + str(ks.reason()) + ')' if ks.active else 'no'}")
    return 0


def cmd_telegram_id(a, settings, secrets) -> int:
    import httpx

    r = httpx.get(f"https://api.telegram.org/bot{secrets.get('TELEGRAM_BOT_TOKEN')}/getUpdates", timeout=20).json()
    chats = {str(u.get("message", {}).get("chat", {}).get("id")): u.get("message", {}).get("chat", {}).get("first_name") for u in r.get("result", []) if "message" in u}
    print("Mandale /start a tu bot y volvé a correr esto." if not chats else "\n".join(f"TELEGRAM_CHAT_ID={k}  ({v})" for k, v in chats.items()))
    return 0


def cmd_rollback(a, settings, secrets) -> int:
    from .brain.nightly import rollback

    prev = rollback(STRATEGY_DIR)
    print(f"Volvimos a {prev.name} v{prev.version}. Reiniciá el bot para aplicarla.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tbot", description="Bot de trading en tres capas (Opus lento, scorer rápido, código que decide).")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backtest", help="walk-forward fuera de muestra y filtro")
    b.add_argument("--synthetic", action="store_true")
    b.add_argument("--years", type=float, default=3.0)
    b.add_argument("--template", default="trend", choices=["trend", "meanrev", "breakout"])
    b.add_argument("--spec")
    r = sub.add_parser("research", help="Opus 5.5 propone estrategias y el arnés las evalúa")
    r.add_argument("--synthetic", action="store_true")
    r.add_argument("--years", type=float, default=3.0)
    r.add_argument("--rounds", type=int, default=3)
    run = sub.add_parser("run", help="corre el bot (paper por defecto)")
    run.add_argument("--observe", action="store_true", help="registra predicciones sin operar")
    run.add_argument("--auto", action="store_true", help="opera con la estrategia aprobada; si no hay, observa con la candidata; si no hay ninguna, espera")
    run.add_argument("--sim", action="store_true", help="simulación con datos sintéticos, sin Alpaca")
    run.add_argument("--days", type=int, default=10)
    d = sub.add_parser("dashboard", help="dashboard en vivo")
    d.add_argument("--host", default="127.0.0.1")
    d.add_argument("--port", type=int, default=8080)
    d.add_argument("--sim", action="store_true")
    k = sub.add_parser("kill", help="activar el kill switch")
    k.add_argument("--reason", default="kill manual")
    sub.add_parser("unkill", help="reactivar después de un kill (pide confirmación)")
    sub.add_parser("kill-drill", help="simulacro real del kill switch en paper")
    sub.add_parser("golive-check", help="chequeo final antes de dinero real")
    rep = sub.add_parser("report", help="reporte diario")
    rep.add_argument("--date")
    sub.add_parser("status")
    sub.add_parser("nightly")
    sub.add_parser("telegram-id", help="muestra tu TELEGRAM_CHAT_ID")
    sub.add_parser("strategy-rollback", help="volver a la versión anterior de la estrategia")
    a = p.parse_args(argv)

    secrets = Secrets.load()
    setup_logging(secrets)
    settings = load_settings()
    fn = {
        "backtest": cmd_backtest, "research": cmd_research, "run": cmd_run, "dashboard": cmd_dashboard, "kill": cmd_kill,
        "unkill": cmd_unkill, "kill-drill": cmd_kill_drill, "golive-check": cmd_golive, "report": cmd_report, "status": cmd_status,
        "nightly": cmd_nightly, "telegram-id": cmd_telegram_id, "strategy-rollback": cmd_rollback,
    }[a.cmd]
    return fn(a, settings, secrets)


if __name__ == "__main__":
    sys.exit(main())
