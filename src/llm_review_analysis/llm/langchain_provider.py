from __future__ import annotations

from typing import Any

from .base import LLMResponse


class LangChainChatProvider:
    """LangChain ChatOpenAI adapter for the paper-consistent live path.

    Imports are lazy so the default mock/test path does not require LangChain
    packages. Construction is gated by ``ALLOW_LIVE_LLM`` because invoking this
    provider can make live API calls.
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
            raise RuntimeError("Live LangChain LLM calls are disabled. Set ALLOW_LIVE_LLM=true only after approval.")
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "LangChain live mode requires the paper-stack dependencies. "
                "Install with: python -m pip install -e \".[paper]\""
            ) from exc

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self._chat_cls = ChatOpenAI
        self._client = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
        )

    def generate(self, prompt: str, *, purpose: str = "general", response_format: str | None = None) -> LLMResponse:
        client = self._client
        if response_format == "json":
            client = self._with_json_response_format()
        response = client.invoke(prompt)
        content = _message_content(response)
        return LLMResponse(content=content, usage=_usage_metadata(response), model=self.model)

    def _with_json_response_format(self) -> Any:
        try:
            return self._client.bind(response_format={"type": "json_object"})
        except Exception:
            return self._chat_cls(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
                max_retries=self.max_retries,
                model_kwargs={"response_format": {"type": "json_object"}},
            )


def _message_content(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "\n".join(str(part) for part in content)
    return str(content or "")


def _usage_metadata(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict):
        return usage
    metadata = getattr(response, "response_metadata", None)
    if isinstance(metadata, dict):
        token_usage = metadata.get("token_usage")
        if isinstance(token_usage, dict):
            return token_usage
    return {}
