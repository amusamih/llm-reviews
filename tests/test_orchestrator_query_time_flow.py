from __future__ import annotations

import sqlite3
from typing import Any

from llm_review_analysis.agents import ReviewOrchestrator
from llm_review_analysis.agents.orchestrator import DirectSQLTrace
from llm_review_analysis.agents.semantic_reasoning_agent import SemanticReasoningTrace
from llm_review_analysis.db.schema import ensure_review_table, insert_review_rows
from llm_review_analysis.llm import LLMResponse


def test_non_english_direct_sql_uses_internal_prompt_and_translates_back(settings, sample_db):
    conn, _ = sample_db
    language = RecordingLanguageAgent(
        language="ar",
        translation="How many reviews for sample product between 2025-07-01 and 2025-07-31?",
    )
    provider = RecordingRouteProvider("DIRECT_SQL")
    orchestrator = ReviewOrchestrator(settings, provider, language_agent=language)

    result, trace = orchestrator.answer_with_trace(conn, "كم عدد مراجعات sample product؟")

    assert result["type"] == "text"
    assert result["message"].startswith("[ar] ")
    assert language.detect_calls == ["كم عدد مراجعات sample product؟"]
    assert language.translate_calls == [("The table contains 2 reviews.", "ar")]
    assert trace["original_prompt"] == "كم عدد مراجعات sample product؟"
    assert trace["original_language"] == "ar"
    assert trace["internal_prompt"] == "How many reviews for sample product between 2025-07-01 and 2025-07-31?"
    assert trace["product_name"] == "sample product"
    assert trace["table"] == "sample_product"
    assert trace["route"] == "DIRECT_SQL"
    assert trace["date_range"] == "2025-07-01..2025-07-31"
    assert "2025-07-01" in trace["sql"]
    assert provider.route_user_requests == [trace["internal_prompt"]]


def test_missing_table_triggers_retrieval_and_uses_returned_table(settings):
    conn = sqlite3.connect(settings.database_path)
    conn.row_factory = sqlite3.Row
    retrieval = RecordingRetrievalAgent(
        conn,
        rows=[
            {
                "asin": "NEW1",
                "rating": "5",
                "title": "Great",
                "content": "The new product works well.",
                "date": "2024-01-01",
            }
        ],
    )
    language = RecordingLanguageAgent(language="en", translation="How many reviews for newly added product?")
    orchestrator = ReviewOrchestrator(
        settings,
        RecordingRouteProvider("DIRECT_SQL"),
        language_agent=language,
        retrieval_agent=retrieval,
    )

    result, trace = orchestrator.answer_with_trace(conn, "How many reviews for newly added product?")

    assert result["message"] == "The table contains 1 reviews."
    assert retrieval.calls == ["newly added product"]
    assert trace["retrieval_attempted"] is True
    assert trace["retrieval_error"] is None
    assert trace["table"] == "newly_added_product"
    assert trace["sql"].startswith("SELECT COUNT(*)")


def test_semantics_uses_internal_prompt_and_translates_textual_answer(settings, sample_db):
    conn, _ = sample_db
    language = RecordingLanguageAgent(
        language="es",
        translation="Why are users unhappy about sample product?",
    )
    semantic = RecordingSemanticAgent("Users mention delivery problems.")
    orchestrator = ReviewOrchestrator(
        settings,
        RecordingRouteProvider("SEMANTICS"),
        language_agent=language,
        semantic_reasoning_agent=semantic,
    )

    result, trace = orchestrator.answer_with_trace(conn, "¿Por qué los usuarios están descontentos con sample product?")

    assert result["message"] == "[es] Users mention delivery problems."
    assert semantic.calls == [("sample_product", "Why are users unhappy about sample product?")]
    assert trace["route"] == "SEMANTICS"
    assert trace["evidence_ids"] == ["7"]
    assert language.translate_calls == [("Users mention delivery problems.", "es")]


def test_analytics_uses_original_prompt_and_matched_table_without_text_translate_back(settings, sample_db):
    conn, _ = sample_db
    original_prompt = "Montrez la distribution des notes pour sample product"
    internal_prompt = "Show the rating distribution for sample product"
    language = RecordingLanguageAgent(language="fr", translation=internal_prompt)
    analytics = RecordingAnalyticsAgent()
    orchestrator = ReviewOrchestrator(
        settings,
        RecordingRouteProvider("ANALYTICS"),
        language_agent=language,
        analytics_agent=analytics,
    )

    result, trace = orchestrator.answer_with_trace(conn, original_prompt)

    assert result["type"] == "chart"
    assert analytics.calls == [("sample_product", original_prompt)]
    assert trace["route"] == "ANALYTICS"
    assert trace["internal_prompt"] == internal_prompt
    assert language.translate_calls == []


def test_controlled_textual_failure_is_translated_back(settings, sample_db):
    conn, _ = sample_db
    language = RecordingLanguageAgent(language="ar", translation="Tell me about sample product.")
    orchestrator = ReviewOrchestrator(settings, RecordingRouteProvider("SEMANTICS"), language_agent=language)

    result, trace = orchestrator.answer_with_trace(conn, "أخبرني عن sample product")

    assert result["type"] == "text"
    assert result["failure_category"] == "ambiguous_prompt"
    assert result["message"].startswith("[ar] ")
    assert trace["failure_category"] == "ambiguous_prompt"
    assert language.translate_calls == [
        (
            "The request is ambiguous. Please specify whether you want a factual count, an evidence-based explanation, or a chart/analytics view.",
            "ar",
        )
    ]


def test_unknown_route_returns_controlled_fallback(settings, sample_db):
    conn, _ = sample_db
    language = RecordingLanguageAgent(language="en", translation="Please process for sample product")
    orchestrator = ReviewOrchestrator(settings, RecordingRouteProvider("UNSUPPORTED_ROUTE"), language_agent=language)

    result, trace = orchestrator.answer_with_trace(conn, "Please process for sample product")

    assert result["type"] == "text"
    assert result["failure_category"] == "unsupported_route"
    assert trace["route"] == "UNSUPPORTED"
    assert trace["failure_category"] == "unsupported_route"
    assert trace["sql"] is None


class RecordingLanguageAgent:
    def __init__(self, *, language: str, translation: str) -> None:
        self.language = language
        self.translation = translation
        self.detect_calls: list[str] = []
        self.translate_calls: list[tuple[str, str]] = []

    def detect_and_translate_text(self, text: str) -> tuple[str, str]:
        self.detect_calls.append(text)
        return self.language, self.translation

    def translate_text(self, text: str, target_language: str) -> str:
        self.translate_calls.append((text, target_language))
        return f"[{target_language}] {text}"


class RecordingRouteProvider:
    model = "recording-route-provider"

    def __init__(self, route_response: str) -> None:
        self.route_response = route_response
        self.route_user_requests: list[str] = []

    def generate(self, prompt: str, *, purpose: str = "general", response_format: str | None = None) -> LLMResponse:
        if purpose == "route":
            self.route_user_requests.append(_extract_user_request(prompt))
            return LLMResponse(content=self.route_response, model=self.model)
        return LLMResponse(content="{}", model=self.model)


class RecordingRetrievalAgent:
    def __init__(self, conn: sqlite3.Connection, *, rows: list[dict[str, Any]]) -> None:
        self.conn = conn
        self.rows = rows
        self.calls: list[str] = []

    def retrieve_live(self, product_name: str) -> str:
        self.calls.append(product_name)
        table = product_name.replace(" ", "_").lower()
        ensure_review_table(self.conn, table)
        insert_review_rows(self.conn, table, self.rows)
        return table


class RecordingSemanticAgent:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls: list[tuple[str, str]] = []

    def answer_with_trace(self, conn: sqlite3.Connection, table_name: str, prompt: str) -> SemanticReasoningTrace:
        self.calls.append((table_name, prompt))
        return SemanticReasoningTrace(
            answer=self.answer,
            evidence_ids=("7",),
            evidence_snippets=("Delivery was late.",),
        )


class RecordingAnalyticsAgent:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def run(self, conn: sqlite3.Connection, table_name: str, prompt: str) -> dict[str, object]:
        self.calls.append((table_name, prompt))
        return {
            "type": "chart",
            "path": "chart.png",
            "chart_type": "bar",
            "aggregation": "count",
            "group_by": "rating",
            "chart_rows": [],
            "explanation": "Chart explanation.",
        }


def _extract_user_request(route_prompt: str) -> str:
    marker = "User request:"
    if marker not in route_prompt:
        return route_prompt
    return route_prompt.rsplit(marker, 1)[-1].split("Route:", 1)[0].strip()
