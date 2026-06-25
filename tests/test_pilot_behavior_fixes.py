from __future__ import annotations

from llm_review_analysis.agents import ReviewOrchestrator
from llm_review_analysis.agents.semantic_reasoning_agent import _format_retrieved_context, _reasoning_prompt


class _Doc:
    def __init__(self, page_content: str, review_id: str) -> None:
        self.page_content = page_content
        self.metadata = {"review_id": review_id}


def test_unknown_product_does_not_fall_back_to_available_table(settings, provider, sample_db):
    conn, _ = sample_db
    orchestrator = ReviewOrchestrator(settings, provider)

    result, trace = orchestrator.answer_with_trace(conn, "How many reviews for unknown gadget?")

    assert result["type"] == "text"
    assert "0 reviews" in result["message"]
    assert trace["table"] is None
    assert trace["failure_category"] == "product_not_found"


def test_ambiguous_prompt_returns_controlled_clarification(settings, provider, sample_db):
    conn, _ = sample_db
    orchestrator = ReviewOrchestrator(settings, provider)

    result, trace = orchestrator.answer_with_trace(conn, "Tell me about sample product.")

    assert result["type"] == "text"
    assert "ambiguous" in result["message"].lower()
    assert trace["route"] == "SEMANTICS"
    assert trace["failure_category"] == "ambiguous_prompt"
    assert trace["evidence_ids"] == []


def test_missing_information_prompt_does_not_fabricate_evidence(settings, provider, sample_db):
    conn, _ = sample_db
    orchestrator = ReviewOrchestrator(settings, provider)

    result, trace = orchestrator.answer_with_trace(conn, "Why is the warranty score for sample product low?")

    assert result["type"] == "text"
    assert "warranty" in result["message"].lower()
    assert "cannot infer" in result["message"].lower()
    assert trace["route"] == "SEMANTICS"
    assert trace["failure_category"] == "missing_information"
    assert trace["evidence_ids"] == []


def test_unknown_product_alias_suffix_is_stripped_only_when_matching_reviews_suffix(settings, provider, sample_db):
    conn, _ = sample_db
    orchestrator = ReviewOrchestrator(settings, provider)

    result, trace = orchestrator.answer_with_trace(conn, "Evaluate translation quality for sample product reviews.")

    assert result["type"] == "text"
    assert trace["table"] == "sample_product"
    assert trace["product_name"] == "sample product"
    assert trace["failure_category"] == "translation_quality_not_evaluated"


def test_unsupported_chart_request_fails_without_rendering_fallback_chart(settings, provider, sample_db):
    conn, _ = sample_db
    orchestrator = ReviewOrchestrator(settings, provider)

    result, trace = orchestrator.answer_with_trace(conn, "Show a scatter plot of rating by date for sample product")

    assert result["type"] == "text"
    assert "unsupported chart type" in result["message"].lower()
    assert trace["route"] == "ANALYTICS"
    assert trace["failure_category"] == "unsupported_chart_type"
    assert trace["chart_path"] is None


def test_semantic_route_logs_source_id_and_snippet_for_exact_phrase(settings, provider, sample_db):
    conn, _ = sample_db
    orchestrator = ReviewOrchestrator(settings, provider)

    result, trace = orchestrator.answer_with_trace(
        conn,
        "Why does review text mention excellent battery life for sample product?",
    )

    evidence_text = " ".join(trace["evidence_snippets"])
    assert result["type"] == "text"
    assert trace["route"] == "SEMANTICS"
    assert "1" in trace["evidence_ids"]
    assert "excellent battery life" in evidence_text.lower()
    assert trace["failure_category"] is None


def test_live_semantic_prompt_foregrounds_exact_phrase_and_source_id():
    prompt = "Why does review text mention with the quality that is for amazon all beauty?"
    context = _format_retrieved_context(
        prompt,
        [
            _Doc(
                "Before filler. The customer was upset with the quality that is not here and wanted a replacement.",
                "15",
            )
        ],
    )
    reasoning_prompt = _reasoning_prompt(prompt, context)

    assert "[Review ID: 15]" in context
    assert "with the quality that is" in context
    assert "Treat product/table names" in reasoning_prompt
    assert "review corpus" in reasoning_prompt
    assert "quote or closely paraphrase" in reasoning_prompt

