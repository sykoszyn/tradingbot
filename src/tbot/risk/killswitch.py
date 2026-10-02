"""Kill switch persistente: si el archivo existe, el bot no abre nada nuevo. Sobrevive reinicios."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

CONFIRM_WORD = "REACTIVAR"


class KillSwitch:
    def __init__(self, state_dir: Path):
        self.path = Path(state_dir) / "KILL"

    @property
    def active(self) -> bool:
        return self.path.exists()

    def reason(self) -> str | None:
        if not self.active:
            return None
        try:
            return json.loads(self.path.read_text()).get("reason")
        except (ValueError, OSError):
            return "desconocido"

    def fire(self, reason: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.active:
            self.path.write_text(json.dumps({"reason": reason, "at": pd.Timestamp.now(tz="UTC").isoformat()}))

    def reset(self, confirmation: str) -> None:
        """Solo a mano, desde la línea de comandos, escribiendo la palabra de confirmación."""
        if confirmation != CONFIRM_WORD:
            raise PermissionError(f"Para reactivar escribí exactamente {CONFIRM_WORD}")
        self.path.unlink(missing_ok=True)
