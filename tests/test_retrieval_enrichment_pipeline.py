from __future__ import annotations

import sqlite3

import pytest

from llm_review_analysis.agents import RetrievalAgent, RetrievalError, ReviewOrchestrator
from llm_review_analysis.llm import LLMResponse


def test_load_records_creates_table_inserts_deduplicates_and_runs_enrichment_in_order(settings):
    conn = sqlite3.connect(settings.database_path)
    conn.row_factory = sqlite3.Row
    calls: list[tuple[str, str, int]] = []
    language = RecordingEnricher("language", calls)
    topic = RecordingEnricher("topic", calls)
    semantic = RecordingEnricher("semantic", calls)
    agent = RetrievalAgent(
        settings,
        language_agent=language,
        topic_agent=topic,
        semantic_tagger=semantic,
    )

    table = agent.load_records(
        conn,
        "Sample Product",
        [
            {
                "asin": "A1",
                "rating": "5",
                "title": "  Great charger  ",
                "content": " Works well ",
                "date": "2025-01-01",
            },
            {
                "asin": "a1",
                "rating": "5",
                "title": "Great charger",
                "content": "works well",
                "date": "2025-01-01",
            },
        ],
    )

    rows = conn.execute(f"SELECT title, content FROM {table}").fetchall()
    assert table == "sample_product"
    assert len(rows) == 1
    assert rows[0]["title"] == "Great charger"
    assert rows[0]["content"] == "Works well"
    assert calls == [
        ("language", "sample_product", 1),
        ("topic", "sample_product", 1),
        ("semantic", "sample_product", 1),
    ]


def test_load_records_returns_table_name_after_successful_enrichment(settings):
    conn = sqlite3.connect(settings.database_path)
    conn.row_factory = sqlite3.Row
    agent = RetrievalAgent(
        settings,
        language_agent=RecordingEnricher("language", []),
        topic_agent=RecordingEnricher("topic", []),
        semantic_tagger=RecordingEnricher("semantic", []),
    )

    table = agent.load_records(conn, "EV Charger", [{"asin": "E1", "title": "Good", "content": "Good product"}])

    assert table == "ev_charger"
    assert conn.execute("SELECT COUNT(*) FROM ev_charger").fetchone()[0] == 1


def test_load_records_empty_input_raises_controlled_retrieval_error(settings):
    conn = sqlite3.connect(settings.database_path)
    agent = RetrievalAgent(
        settings,
        language_agent=RecordingEnricher("language", []),
        topic_agent=RecordingEnricher("topic", []),
        semantic_tagger=RecordingEnricher("semantic", []),
    )

    with pytest.raises(RetrievalError, match="No review records"):
        agent.load_records(conn, "Missing Product", [])


def test_load_records_enrichment_failure_raises_controlled_retrieval_error(settings):
    conn = sqlite3.connect(settings.database_path)
    conn.row_factory = sqlite3.Row
    agent = RetrievalAgent(
        settings,
        language_agent=RecordingEnricher("language", []),
        topic_agent=FailingEnricher(),
        semantic_tagger=RecordingEnricher("semantic", []),
    )

    with pytest.raises(RetrievalError, match="Review enrichment failed"):
        agent.load_records(conn, "Sample Product", [{"asin": "A1", "title": "Good", "content": "Good product"}])


def test_orchestrator_reports_retrieval_failure_as_controlled_failure(settings):
    conn = sqlite3.connect(settings.database_path)
    conn.row_factory = sqlite3.Row
    orchestrator = ReviewOrchestrator(
        settings,
        StaticRouteProvider("DIRECT_SQL"),
        language_agent=StaticLanguageAgent("en", "How many reviews for missing product?"),
        retrieval_agent=FailingRetrievalAgent(),
    )

    result, trace = orchestrator.answer_with_trace(conn, "How many reviews for missing product?")

    assert result["type"] == "text"
    assert result["failure_category"] == "retrieval_failed"
    assert trace["retrieval_attempted"] is True
    assert trace["retrieval_error"] == "fixture retrieval failed"
    assert trace["failure_category"] == "retrieval_failed"
    assert trace["route"] is None


class RecordingEnricher:
    def __init__(self, name: str, calls: list[tuple[str, str, int]]) -> None:
        self.name = name
        self.calls = calls

    def enrich_table(self, conn: sqlite3.Connection, table_name: str) -> int:
        count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        self.calls.append((self.name, table_name, count))
        return count


class FailingEnricher:
    def enrich_table(self, conn: sqlite3.Connection, table_name: str) -> int:
        raise RuntimeError("topic service unavailable")


class FailingRetrievalAgent:
    def retrieve_live(self, product_name: str) -> str:
        raise RetrievalError("fixture retrieval failed")


class StaticLanguageAgent:
    def __init__(self, language: str, translation: str) -> None:
        self.language = language
        self.translation = translation

    def detect_and_translate_text(self, text: str) -> tuple[str, str]:
        return self.language, self.translation

    def translate_text(self, text: str, target_language: str) -> str:
        return text


class StaticRouteProvider:
    model = "static-route-provider"

    def __init__(self, route: str) -> None:
        self.route = route

    def generate(self, prompt: str, *, purpose: str = "general", response_format: str | None = None) -> LLMResponse:
        return LLMResponse(content=self.route if purpose == "route" else "{}", model=self.model)
