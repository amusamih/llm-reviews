import pytest

from llm_review_analysis.agents import SemanticReasoningAgent, expected_langchain_tool_names
from llm_review_analysis.config import load_settings
from llm_review_analysis.llm import LangChainChatProvider, MockLLMProvider
from llm_review_analysis.providers import build_llm_provider


def test_langchain_provider_is_gated_before_optional_imports():
    with pytest.raises(RuntimeError, match="Live LangChain LLM calls are disabled"):
        LangChainChatProvider("gpt-4o", allow_live=False)


def test_build_provider_supports_langchain_alias_but_requires_live_approval():
    settings = load_settings({"LLM_PROVIDER": "langchain"})
    with pytest.raises(RuntimeError, match="Live LangChain LLM calls are disabled"):
        build_llm_provider(settings)


def test_langchain_tool_contract_names_are_available_without_langchain():
    assert expected_langchain_tool_names() == (
        "route_prompt",
        "direct_sql_query",
        "semantic_reasoning",
        "data_analytics",
        "data_retrieval",
        "language_translation",
        "topic_inference",
        "topic_assignment",
        "semantic_tagging",
    )


def test_faiss_semantic_reasoning_requires_live_approval(settings, sample_db):
    conn, table = sample_db
    agent = SemanticReasoningAgent(
        settings=settings,
        provider=MockLLMProvider(),
        backend="faiss",
    )

    with pytest.raises(RuntimeError, match="ALLOW_LIVE_LLM=true"):
        agent.answer(conn, table, "Why are users unhappy?")
