from __future__ import annotations

from .base import LLMResponse
from .prompt_formatting import user_chat_messages


class OpenAILLMProvider:
    """OpenAI provider wrapper.

    Instantiation is allowed only when live calls have been explicitly approved
    through configuration. The import is lazy so tests can run without the
    OpenAI package installed.
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
            raise RuntimeError("Live LLM calls are disabled. Set ALLOW_LIVE_LLM=true only after approval.")
        from openai import OpenAI

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = OpenAI(timeout=timeout, max_retries=max_retries)

    def generate(self, prompt: str, *, purpose: str = "general", response_format: str | None = None) -> LLMResponse:
        kwargs = {
            "model": self.model,
            "messages": user_chat_messages(prompt, response_format=response_format),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}
        response = self._client.chat.completions.create(**kwargs)
        message = response.choices[0].message.content or ""
        usage = response.usage.model_dump() if getattr(response, "usage", None) else {}
        return LLMResponse(content=message, usage=usage, model=self.model)
