from __future__ import annotations

import sqlite3

from llm_review_analysis.agents import ConversationState, ReviewOrchestrator
from llm_review_analysis.agents.analytics_agent import build_aggregate_sql
from llm_review_analysis.agents.semantic_reasoning_agent import SemanticReasoningTrace
from llm_review_analysis.analytics import ChartSpec
from llm_review_analysis.db.schema import REVIEW_COLUMNS, ensure_review_table, insert_review_rows
from llm_review_analysis.db.sql_validator import validate_select_sql
from llm_review_analysis.llm import LLMResponse


def test_stateful_followup_inherits_product_and_applies_rating_filter(settings, sample_db):
    conn, _ = sample_db
    provider = SequencedRouteProvider(["DIRECT_SQL", "DIRECT_SQL"])
    orchestrator = ReviewOrchestrator(settings, provider)

    first, state, first_trace = orchestrator.answer_with_state(conn, "How many reviews for sample product?")
    second, state, second_trace = orchestrator.answer_with_state(
        conn,
        "Now do the same for five-star reviews.",
        state=state,
    )

    assert first["message"] == "The table contains 2 reviews."
    assert first_trace["table"] == "sample_product"
    assert second["message"] == "The table contains 1 reviews."
    assert second_trace["state_context_used"] is True
    assert second_trace["context_dependent"] is True
    assert second_trace["table"] == "sample_product"
    assert "sample product" in second_trace["internal_prompt"].lower()
    assert "CAST(rating AS REAL) = 5" in second_trace["sql"]
    assert state.matched_table == "sample_product"
    assert state.active_filters == {"rating": "5"}



def test_stateful_followup_resolves_omitted_product_without_pronoun(settings, sample_db):
    conn, _ = sample_db
    orchestrator = ReviewOrchestrator(settings, SequencedRouteProvider(["DIRECT_SQL", "DIRECT_SQL"]))

    _, state, _ = orchestrator.answer_with_state(conn, "How many reviews for sample product?")
    result, state, trace = orchestrator.answer_with_state(conn, "What is the average rating?", state=state)

    assert result["message"] == "The average rating is 3.50."
    assert trace["state_context_used"] is True
    assert trace["context_dependent"] is True
    assert trace["table"] == "sample_product"
    assert "sample product" in trace["internal_prompt"].lower()
    assert state.previous_numeric_result == 3.5

def test_stateful_semantic_followup_uses_prior_evidence(settings, sample_db):
    conn, _ = sample_db
    semantic = RecordingSemanticAgent(
        answers=["Users mention delivery problems.", "The follow-up is grounded in the prior delivery evidence."],
        snippets=[("Delivery was late.",), ("Delivery was late.",)],
    )
    orchestrator = ReviewOrchestrator(
        settings,
        SequencedRouteProvider(["SEMANTICS", "SEMANTICS"]),
        semantic_reasoning_agent=semantic,
    )

    _, state, first_trace = orchestrator.answer_with_state(conn, "Why are users unhappy about sample product?")
    result, state, second_trace = orchestrator.answer_with_state(conn, "Why do users say that?", state=state)

    assert first_trace["evidence_snippets"] == ["Delivery was late."]
    assert result["message"] == "The follow-up is grounded in the prior delivery evidence."
    assert second_trace["failure_category"] is None
    assert second_trace["state_context_used"] is True
    assert "sample product" in semantic.calls[-1][1].lower()
    assert "delivery was late" in semantic.calls[-1][1].lower()
    assert state.evidence_summary == "Delivery was late."


def test_stateful_analytics_followup_receives_resolved_prompt(settings, sample_db):
    conn, _ = sample_db
    analytics = RecordingAnalyticsAgent()
    orchestrator = ReviewOrchestrator(
        settings,
        SequencedRouteProvider(["DIRECT_SQL", "ANALYTICS"]),
        analytics_agent=analytics,
    )

    _, state, _ = orchestrator.answer_with_state(conn, "What is the average rating for sample product?")
    result, state, trace = orchestrator.answer_with_state(conn, "Show that by month.", state=state)

    assert result["type"] == "chart"
    assert analytics.calls[-1][0] == "sample_product"
    assert analytics.calls[-1][1] == trace["analytics_prompt"]
    assert "average rating by date" in analytics.calls[-1][1].lower()
    assert "sample product" in analytics.calls[-1][1].lower()
    assert state.previous_chart_spec == {"chart_type": "line", "group_by": "date", "aggregation": "avg"}


def test_stateful_first_turn_analytics_preserves_original_prompt(settings, sample_db):
    conn, _ = sample_db
    original_prompt = "Montrez la distribution des notes pour sample product"
    internal_prompt = "Show the rating distribution for sample product"
    language = RecordingLanguageAgent(language="fr", translation=internal_prompt)
    analytics = RecordingAnalyticsAgent()
    orchestrator = ReviewOrchestrator(
        settings,
        SequencedRouteProvider(["ANALYTICS"]),
        language_agent=language,
        analytics_agent=analytics,
    )

    _, state, trace = orchestrator.answer_with_state(conn, original_prompt)

    assert analytics.calls == [("sample_product", original_prompt)]
    assert trace["internal_prompt"] == internal_prompt
    assert trace["analytics_prompt"] == original_prompt
    assert state.response_language == "fr"


def test_state_reset_clears_context_and_followup_requires_clarification(settings, sample_db):
    conn, _ = sample_db
    orchestrator = ReviewOrchestrator(settings, SequencedRouteProvider(["DIRECT_SQL", "DIRECT_SQL"]))

    _, state, _ = orchestrator.answer_with_state(conn, "How many reviews for sample product?")
    reset_result, reset_state, reset_trace = orchestrator.answer_with_state(conn, "reset context", state=state)
    followup, _, followup_trace = orchestrator.answer_with_state(conn, "What about only negative reviews?", state=reset_state)

    assert reset_result["message"] == "Conversation context has been reset."
    assert reset_state == ConversationState.reset()
    assert reset_trace["state_reset"] is True
    assert followup["failure_category"] == "context_missing"
    assert followup_trace["failure_category"] == "context_missing"


def test_state_isolated_between_sessions(settings, sample_db):
    conn, _ = sample_db
    orchestrator = ReviewOrchestrator(settings, SequencedRouteProvider(["DIRECT_SQL", "DIRECT_SQL"]))

    _, state, _ = orchestrator.answer_with_state(conn, "How many reviews for sample product?")
    isolated_result, isolated_state, trace = orchestrator.answer_with_state(conn, "Now do the same for five-star reviews.")

    assert state.matched_table == "sample_product"
    assert isolated_result["failure_category"] == "context_missing"
    assert isolated_state.matched_table is None
    assert trace["state_context_used"] is False


def test_product_switch_does_not_carry_prior_filters(settings, sample_db):
    conn, _ = sample_db
    ensure_review_table(conn, "other_product")
    insert_review_rows(
        conn,
        "other_product",
        [
            {"asin": "OTHER1", "rating": "1", "title": "Other", "content": "Other product review.", "date": "2025-08-01"},
            {"asin": "OTHER2", "rating": "2", "title": "Other", "content": "Another other product review.", "date": "2025-08-02"},
        ],
    )
    orchestrator = ReviewOrchestrator(settings, SequencedRouteProvider(["DIRECT_SQL", "DIRECT_SQL", "DIRECT_SQL"]))

    _, state, _ = orchestrator.answer_with_state(conn, "How many reviews for sample product?")
    _, state, _ = orchestrator.answer_with_state(conn, "Now do the same for five-star reviews.", state=state)
    result, switched_state, trace = orchestrator.answer_with_state(conn, "How many reviews for other product?", state=state)

    assert result["message"] == "The table contains 2 reviews."
    assert trace["product_switched"] is True
    assert "CAST(rating AS REAL) = 5" not in trace["sql"]
    assert switched_state.matched_table == "other_product"
    assert switched_state.active_filters == {}


def test_stale_context_returns_controlled_failure(settings, sample_db):
    conn, _ = sample_db
    stale_state = ConversationState(product_name="sample product", matched_table="sample_product", turn_count=8)
    orchestrator = ReviewOrchestrator(settings, SequencedRouteProvider(["DIRECT_SQL"]))

    result, updated, trace = orchestrator.answer_with_state(conn, "What about only negative reviews?", state=stale_state)

    assert result["failure_category"] == "stale_context"
    assert trace["failure_category"] == "stale_context"
    assert updated.matched_table == "sample_product"


def test_analytics_sql_uses_safe_filters_from_resolved_prompt():
    spec = ChartSpec.from_mapping(
        {
            "chart_type": "line",
            "x_field": "date",
            "y_field": "rating",
            "aggregation": "avg",
            "group_by": "date",
            "title": "Average rating over time",
        }
    )

    sql = build_aggregate_sql(
        "sample_product",
        spec,
        prompt="Show average rating by date for sample product with five-star negative reviews",
    )

    assert "WHERE" in sql
    assert "CAST(rating AS REAL) = 5" in sql
    assert "LOWER(semantic_tags) LIKE '%negative%'" in sql
    validate_select_sql(sql, allowed_tables=["sample_product"], allowed_columns=REVIEW_COLUMNS)


class SequencedRouteProvider:
    model = "sequenced-route-provider"

    def __init__(self, routes: list[str]) -> None:
        self.routes = list(routes)
        self.route_user_requests: list[str] = []

    def generate(self, prompt: str, *, purpose: str = "general", response_format: str | None = None) -> LLMResponse:
        if purpose == "route":
            self.route_user_requests.append(_extract_user_request(prompt))
            route = self.routes.pop(0) if self.routes else "DIRECT_SQL"
            return LLMResponse(content=route, model=self.model)
        return LLMResponse(content="{}", model=self.model)


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


class RecordingSemanticAgent:
    def __init__(self, *, answers: list[str], snippets: list[tuple[str, ...]]) -> None:
        self.answers = list(answers)
        self.snippets = list(snippets)
        self.calls: list[tuple[str, str]] = []

    def answer_with_trace(self, conn: sqlite3.Connection, table_name: str, prompt: str) -> SemanticReasoningTrace:
        self.calls.append((table_name, prompt))
        answer = self.answers.pop(0)
        snippets = self.snippets.pop(0)
        return SemanticReasoningTrace(answer=answer, evidence_ids=("7",), evidence_snippets=snippets)


class RecordingAnalyticsAgent:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def run(self, conn: sqlite3.Connection, table_name: str, prompt: str) -> dict[str, object]:
        self.calls.append((table_name, prompt))
        return {
            "type": "chart",
            "path": "chart.png",
            "chart_type": "line",
            "aggregation": "avg",
            "group_by": "date",
            "chart_rows": [],
            "explanation": "Chart explanation.",
        }


def _extract_user_request(route_prompt: str) -> str:
    marker = "User request:"
    if marker not in route_prompt:
        return route_prompt
    return route_prompt.rsplit(marker, 1)[-1].split("Route:", 1)[0].strip()