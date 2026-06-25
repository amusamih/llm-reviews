from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import platform


@dataclass(frozen=True)
class RunManifest:
    command: str
    dataset: str
    live_mode: bool
    model_provider: str
    repetitions: int

    def to_json(self) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "command": self.command,
            "dataset": self.dataset,
            "live_mode": self.live_mode,
            "model_provider": self.model_provider,
            "repetitions": self.repetitions,
            "platform": platform.platform(),
            "python": platform.python_version(),
        }
        return json.dumps(payload, indent=2)


def write_manifest(path: str | Path, manifest: RunManifest) -> None:
    Path(path).write_text(manifest.to_json() + "\n", encoding="utf-8")
