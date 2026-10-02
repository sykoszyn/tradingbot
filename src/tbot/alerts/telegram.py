"""Bot de Telegram (creado con @BotFather): avisos, aprobaciones con botones, /kill y /status.

Solo obedece al chat configurado (TELEGRAM_CHAT_ID). Los mensajes de otros chats se ignoran.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

log = logging.getLogger(__name__)
API = "https://api.telegram.org/bot{token}/{method}"


class TelegramBot:
    def __init__(
        self,
        *,
        token: str,
        chat_id: str,
        on_approve: Callable[[str], object],
        on_reject: Callable[[str], object],
        on_kill: Callable[[str], object],
        on_status: Callable[[], str],
        http=None,
    ):
        self.token, self.chat_id = token, str(chat_id)
        self.on_approve, self.on_reject, self.on_kill, self.on_status = on_approve, on_reject, on_kill, on_status
        self.http = http
        self.offset = 0

    def _call(self, method: str, **payload):
        if self.http is None:
            return None
        try:
            r = self.http.post(API.format(token=self.token, method=method), json=payload, timeout=35)
            return r.json()
        except Exception as e:  # la falta de Telegram nunca frena el trading
            log.error("Telegram %s falló: %s", method, type(e).__name__)
            return None

    def send(self, text: str) -> None:
        self._call("sendMessage", chat_id=self.chat_id, text=text[:4000])

    def send_approval(self, pending_id: str, text: str) -> None:
        kb = {"inline_keyboard": [[{"text": "✅ Aprobar", "callback_data": f"approve:{pending_id}"}, {"text": "❌ Rechazar", "callback_data": f"reject:{pending_id}"}]]}
        self._call("sendMessage", chat_id=self.chat_id, text="🙋 " + text[:3900], reply_markup=kb)

    def handle_update(self, u: dict) -> None:
        if "callback_query" in u:
            cq = u["callback_query"]
            if str(cq.get("message", {}).get("chat", {}).get("id")) != self.chat_id:
                return
            action, _, pid = (cq.get("data") or "").partition(":")
            self._call("answerCallbackQuery", callback_query_id=cq.get("id"))
            if action == "approve":
                self.on_approve(pid)
            elif action == "reject":
                self.on_reject(pid)
                self.send(f"Rechazada {pid}")
            return
        msg = u.get("message") or {}
        if str(msg.get("chat", {}).get("id")) != self.chat_id:
            return
        text = (msg.get("text") or "").strip()
        if text.startswith("/kill"):
            reason = text[5:].strip() or "kill manual desde Telegram"
            self.on_kill(reason)
        elif text.startswith("/status"):
            self.send(self.on_status())
        elif text.startswith("/start") or text.startswith("/help"):
            self.send("Comandos: /status · /kill [motivo]. Reactivar después de un kill solo se puede desde el servidor (tbot unkill).")

    def poll_once(self, timeout: int = 25) -> None:
        res = self._call("getUpdates", offset=self.offset, timeout=timeout, allowed_updates=["message", "callback_query"])
        for u in (res or {}).get("result", []):
            self.offset = max(self.offset, u["update_id"] + 1)
            try:
                self.handle_update(u)
            except Exception:
                log.exception("Error manejando un update de Telegram")


class TelegramNotifier:
    def __init__(self, bot: TelegramBot):
        self.bot = bot

    def notify(self, kind: str, text: str) -> None:
        log.info("[%s] %s", kind, text)
        self.bot.send(text)

    def request_approval(self, pending_id: str, text: str) -> None:
        self.bot.send_approval(pending_id, text)
