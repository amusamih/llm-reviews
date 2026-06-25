from __future__ import annotations

from typing import Any

from .base import LLMResponse
from .prompt_formatting import adapted_prompt, text_from_content_blocks


class AnthropicLLMProvider:
    """Anthropic Messages API adapter for model-substitution runs.

    Construction is gated by ``ALLOW_LIVE_LLM`` and imports are lazy so the
    offline/test path does not require the Anthropic SDK.
    """

    def __init__(
        self,
        model: str,
        *,
        allow_live: bool,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        if not allow_live:
            raise RuntimeError("Live Anthropic LLM calls are disabled. Set ALLOW_LIVE_LLM=true only after approval.")
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise RuntimeError(
                "Anthropic model-substitution mode requires the Anthropic SDK. "
                "Install with: python -m pip install -e \".[model-substitution]\""
            ) from exc

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = Anthropic(timeout=timeout, max_retries=max_retries)

    def generate(self, prompt: str, *, purpose: str = "general", response_format: str | None = None) -> LLMResponse:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": adapted_prompt(prompt, response_format=response_format)}],
        )
        return LLMResponse(
            content=text_from_content_blocks(getattr(response, "content", "")),
            usage=_anthropic_usage(response),
            model=self.model,
        )


def _anthropic_usage(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return {
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "total_tokens": _sum_tokens(usage.get("input_tokens"), usage.get("output_tokens")),
        }
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": _sum_tokens(input_tokens, output_tokens),
    }


def _sum_tokens(input_tokens: Any, output_tokens: Any) -> int | None:
    if isinstance(input_tokens, (int, float)) and isinstance(output_tokens, (int, float)):
        return int(input_tokens) + int(output_tokens)
    return None
