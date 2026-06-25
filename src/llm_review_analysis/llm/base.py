from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class LLMResponse:
    content: str
    usage: dict[str, Any] = field(default_factory=dict)
    model: str | None = None


class LLMProvider(Protocol):
    """Minimal provider interface used by agents and tests."""

    def generate(self, prompt: str, *, purpose: str = "general", response_format: str | None = None) -> LLMResponse:
        ...
