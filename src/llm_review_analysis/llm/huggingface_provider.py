from __future__ import annotations

import os
import re
from typing import Any

from .base import LLMResponse
from .prompt_formatting import user_chat_messages


class HuggingFaceEndpointProvider:
    """Hugging Face inference endpoint adapter for open-weight comparisons."""

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
            raise RuntimeError("Live Hugging Face endpoint calls are disabled. Set ALLOW_LIVE_LLM=true only after approval.")
        try:
            from huggingface_hub import InferenceClient
        except ImportError as exc:
            raise RuntimeError(
                "Hugging Face model-substitution mode requires huggingface_hub. "
                "Install with: python -m pip install -e \".[model-substitution]\""
            ) from exc

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN or HUGGINGFACEHUB_API_TOKEN is required for Hugging Face endpoint calls.")

        endpoint_url = (os.environ.get("HF_ENDPOINT_URL") or "").strip()
        self.model = model
        self.endpoint_url_configured = bool(endpoint_url)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        client_kwargs = {"token": token, "timeout": timeout}
        if endpoint_url:
            client_kwargs["base_url"] = endpoint_url
        else:
            client_kwargs["model"] = model
        self._client = InferenceClient(**client_kwargs)

    def generate(self, prompt: str, *, purpose: str = "general", response_format: str | None = None) -> LLMResponse:
        messages = user_chat_messages(prompt, response_format=response_format)
        response = _chat_completion(
            self._client,
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return LLMResponse(content=_hf_message_text(response), usage=_hf_usage(response), model=self.model)


def _chat_completion(client: Any, *, model: str, messages: list[dict[str, str]], max_tokens: int, temperature: float) -> Any:
    if hasattr(client, "chat_completion"):
        try:
            return client.chat_completion(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            raise RuntimeError(_redact_hf_error(str(exc))) from exc
    try:
        return client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception as exc:
        raise RuntimeError(_redact_hf_error(str(exc))) from exc


def _hf_message_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    choices = getattr(response, "choices", None)
    if not choices and isinstance(response, dict):
        choices = response.get("choices")
    if choices:
        first = choices[0]
        message = getattr(first, "message", None)
        if message is None and isinstance(first, dict):
            message = first.get("message")
        if isinstance(message, dict):
            return str(message.get("content") or "")
        content = getattr(message, "content", None)
        if content is not None:
            return str(content)
        text = getattr(first, "text", None)
        if text is not None:
            return str(text)
    generated_text = getattr(response, "generated_text", None)
    if generated_text is not None:
        return str(generated_text)
    return str(response or "")


def _hf_usage(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return {
            "input_tokens": usage.get("prompt_tokens") or usage.get("input_tokens"),
            "output_tokens": usage.get("completion_tokens") or usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }
    return {
        "input_tokens": getattr(usage, "prompt_tokens", getattr(usage, "input_tokens", None)),
        "output_tokens": getattr(usage, "completion_tokens", getattr(usage, "output_tokens", None)),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def _redact_hf_error(text: str) -> str:
    redacted = str(text)
    for key in ("HF_ENDPOINT_URL", "HF_LLAMA_ENDPOINT_URL", "HF_QWEN_ENDPOINT_URL", "HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        value = os.environ.get(key)
        if value:
            label = "<redacted endpoint URL>" if key.endswith("ENDPOINT_URL") else "<redacted secret>"
            redacted = redacted.replace(value, label)
    redacted = re.sub(
        r"https://[A-Za-z0-9._/-]*endpoints\.huggingface\.cloud(?:/[^\s'\"}]*)?",
        "<redacted endpoint URL>",
        redacted,
    )
    redacted = re.sub(r"\b(?:hf|sk)-[A-Za-z0-9_\-]{20,}\b", "<redacted secret>", redacted)
    return redacted
