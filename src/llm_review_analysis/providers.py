from __future__ import annotations

from .config import Settings
from .llm import (
    AnthropicLLMProvider,
    HuggingFaceEndpointProvider,
    LLMProvider,
    LangChainChatProvider,
    MockLLMProvider,
    OpenAILLMProvider,
)


def build_llm_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "mock":
        return MockLLMProvider()
    if settings.llm_provider in {"langchain", "langchain-openai", "langchain_openai"}:
        return LangChainChatProvider(
            settings.llm_model,
            allow_live=settings.allow_live_llm,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
    if settings.llm_provider == "openai":
        return OpenAILLMProvider(
            settings.llm_model,
            allow_live=settings.allow_live_llm,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
    if settings.llm_provider in {"anthropic", "claude"}:
        return AnthropicLLMProvider(
            settings.llm_model,
            allow_live=settings.allow_live_llm,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
    if settings.llm_provider in {"huggingface", "huggingface_endpoint", "hf"}:
        return HuggingFaceEndpointProvider(
            settings.llm_model,
            allow_live=settings.allow_live_llm,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
    raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")
