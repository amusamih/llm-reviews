from __future__ import annotations

import json
import sqlite3
from typing import Any

from llm_review_analysis.agents.analytics_agent import AnalyticsAgent
from llm_review_analysis.agents.language_agent import LanguageAgent
from llm_review_analysis.agents.retrieval_agent import RetrievalAgent
from llm_review_analysis.agents.semantic_reasoning_agent import SemanticReasoningAgent
from llm_review_analysis.agents.semantic_tagger import SemanticTagger
from llm_review_analysis.agents.topic_assignment_agent import TopicAssignmentAgent
from llm_review_analysis.config import Settings
from llm_review_analysis.llm import LLMProvider


LANGCHAIN_TOOL_NAMES: tuple[str, ...] = (
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


def expected_langchain_tool_names() -> tuple[str, ...]:
    return LANGCHAIN_TOOL_NAMES


def build_langchain_tools(
    settings: Settings,
    provider: LLMProvider,
    conn: sqlite3.Connection,
    table_name: str,
    *,
    orchestrator: Any | None = None,
) -> list[Any]:
    """Build LangChain tool wrappers around the implementation components.

    This function is intentionally lazy-imported so the default mock/test path
    does not require LangChain packages. The returned objects are suitable for
    the paper-consistent live orchestration path.
    """

    StructuredTool = _load_structured_tool()
    from llm_review_analysis.agents.orchestrator import ReviewOrchestrator

    review_orchestrator = orchestrator or ReviewOrchestrator(settings, provider)
    analytics = AnalyticsAgent(settings, provider)
    semantics = SemanticReasoningAgent(settings=settings, provider=provider, backend=settings.semantic_retrieval_backend)
    language = LanguageAgent(provider)
    topics = TopicAssignmentAgent(provider)
    tagger = SemanticTagger(provider=provider, use_provider=True)
    retrieval = RetrievalAgent(settings, provider=provider)

    def route_prompt(question: str) -> str:
        return review_orchestrator.route(question)

    def direct_sql_query(question: str) -> str:
        return review_orchestrator._answer_direct_sql(conn, table_name, question)

    def semantic_reasoning(question: str) -> str:
        return semantics.answer(conn, table_name, question)

    def data_analytics(question: str) -> str:
        result = analytics.run(conn, table_name, question)
        return json.dumps({key: value for key, value in result.items() if key != "base64"}, sort_keys=True)

    def data_retrieval(product_name: str) -> str:
        return retrieval.retrieve_live(product_name)

    def language_translation(text: str) -> str:
        language_code, translation = language.detect_and_translate_text(text)
        return json.dumps({"language": language_code, "translation": translation}, sort_keys=True)

    def topic_inference(review_texts: str) -> str:
        inferred = topics.infer_topics([line for line in review_texts.splitlines() if line.strip()])
        return json.dumps({"topics": inferred}, sort_keys=True)

    def topic_assignment(review_text: str, topics_csv: str) -> str:
        allowed_topics = [topic.strip().lower() for topic in topics_csv.split(",") if topic.strip()]
        assigned = topics.assign_topic(review_text, allowed_topics or list(topics.infer_topics([review_text])))
        return json.dumps({"topic": assigned}, sort_keys=True)

    def semantic_tagging(review_text: str) -> str:
        return json.dumps({"semantic_tags": tagger.tag_text(review_text)}, sort_keys=True)

    tool_specs = [
        (route_prompt, "route_prompt", "Choose DIRECT_SQL, SEMANTICS, or ANALYTICS for a user question."),
        (direct_sql_query, "direct_sql_query", "Answer a factual review question using validated SELECT-only SQL."),
        (semantic_reasoning, "semantic_reasoning", "Answer interpretive questions with review evidence and semantic retrieval."),
        (data_analytics, "data_analytics", "Generate a validated chart specification, render a chart, and explain it."),
        (data_retrieval, "data_retrieval", "Retrieve and store review data through an approved live retrieval adapter."),
        (language_translation, "language_translation", "Detect review language and translate to English if needed."),
        (topic_inference, "topic_inference", "Infer review topics from sample review texts."),
        (topic_assignment, "topic_assignment", "Assign a review to one of the provided topics."),
        (semantic_tagging, "semantic_tagging", "Assign multi-label semantic tags to a review."),
    ]
    return [
        StructuredTool.from_function(func=func, name=name, description=description)
        for func, name, description in tool_specs
    ]


def _load_structured_tool() -> Any:
    try:
        from langchain_core.tools import StructuredTool

        return StructuredTool
    except ImportError as exc:
        raise RuntimeError(
            "LangChain tool wrappers require paper-stack dependencies. "
            "Install with: python -m pip install -e \".[paper]\""
        ) from exc
