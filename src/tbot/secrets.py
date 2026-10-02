"""Claves: solo desde .env / variables de entorno, y nunca visibles en logs."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping

from dotenv import dotenv_values

from .config import ROOT

SECRET_KEYS = ("ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ANTHROPIC_API_KEY", "TELEGRAM_BOT_TOKEN", "DASHBOARD_PASSWORD")


class Secrets:
    def __init__(self, values: Mapping[str, str | None]):
        self._values = {k: v for k, v in values.items() if v}

    @classmethod
    def load(cls) -> Secrets:
        file_values = dotenv_values(ROOT / ".env") if (ROOT / ".env").exists() else {}
        merged = {**file_values, **{k: os.environ[k] for k in os.environ if k in SECRET_KEYS or k.startswith("TELEGRAM_")}}
        return cls(merged)

    def get(self, key: str) -> str:
        if key not in self._values:
            raise KeyError(f"Falta {key} en .env")
        return self._values[key]

    def optional(self, key: str) -> str | None:
        return self._values.get(key)

    def sensitive_values(self) -> list[str]:
        return [v for k, v in self._values.items() if k in SECRET_KEYS and len(v) >= 6]

    def __repr__(self) -> str:
        return f"Secrets({sorted(self._values)})"


class RedactingFilter(logging.Filter):
    """Reemplaza cualquier clave conocida por *** en el mensaje final del log."""

    def __init__(self, secrets: Secrets):
        super().__init__()
        self._needles = sorted(secrets.sensitive_values(), key=len, reverse=True)

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        for n in self._needles:
            msg = msg.replace(n, "***")
        record.msg, record.args = msg, None
        return True


def setup_logging(secrets: Secrets, level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.addFilter(RedactingFilter(secrets))
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
