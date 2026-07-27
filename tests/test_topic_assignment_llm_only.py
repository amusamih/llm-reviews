from __future__ import annotations

import sqlite3

import pytest

from llm_review_analysis.agents.analysis_text import build_analysis_text


def _paired_original_text(row: dict[str, object]) -> str:
    return str(row.get("content") or "").strip()


def _paired_translated_text(row: dict[str, object]) -> str:
    return str(row.get("translated_review") or "").strip()
from llm_review_analysis.agents.topic_assignment_agent import TopicAssignmentAgent, TopicAssignmentError, _parse_assigned_topic
from llm_review_analysis.db.schema import ensure_review_table, insert_review_rows
from llm_review_analysis.llm import LLMResponse


TOPICS = ["design", "battery life", "delivery", "price", "usability", "support"]


class ScriptedProvider:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, str | None]] = []

    def generate(self, prompt: str, *, purpose: str = "general", response_format: str | None = None) -> LLMResponse:
        self.calls.append({"prompt": prompt, "purpose": purpose, "response_format": response_format})
        if self.responses:
            return LLMResponse(content=self.responses.pop(0), model="scripted")
        return LLMResponse(content='{"topic": "support"}', model="scripted")


def test_no_substring_shortcut_when_exact_topic_phrase_is_not_primary() -> None:
    provider = ScriptedProvider(['{"topic": "battery life"}'])
    agent = TopicAssignmentAgent(provider)

    topic = agent.assign_topic("The design is attractive, but the battery life makes the phone unusable.", TOPICS)

    assert topic == "battery life"
    assert len(provider.calls) == 1
    assert "Do not select a topic merely because its words appear in the text" in provider.calls[0]["prompt"]


def test_review_without_exact_topic_phrase_still_uses_provider() -> None:
    provider = ScriptedProvider(['{"topic": "delivery"}'])
    agent = TopicAssignmentAgent(provider)

    assert agent.assign_topic("It arrived late and the box was crushed.", TOPICS) == "delivery"
    assert len(provider.calls) == 1


def test_multiple_topics_and_late_main_complaint_are_llm_decisions() -> None:
    provider = ScriptedProvider(['{"topic": "battery life"}'])
    agent = TopicAssignmentAgent(provider)

    topic = agent.assign_topic("The screen is vivid and setup was easy. After two hours, the charge was gone.", TOPICS)

    assert topic == "battery life"
    assert provider.calls[0]["purpose"] == "topic_assign"


def test_negated_topic_mention_does_not_trigger_local_label_presence_decision() -> None:
    provider = ScriptedProvider(['{"topic": "price"}'])
    agent = TopicAssignmentAgent(provider)

    assert agent.assign_topic("This is not a delivery problem; the real issue is the price.", TOPICS) == "price"
    assert len(provider.calls) == 1


def test_multilingual_review_text_is_sent_to_provider() -> None:
    provider = ScriptedProvider(['{"topic": "battery life"}'])
    agent = TopicAssignmentAgent(provider)

    assert agent.assign_topic("La bateria se agota antes del mediodia.", TOPICS) == "battery life"
    assert "La bateria" in provider.calls[0]["prompt"]


def test_malformed_json_and_out_of_vocabulary_outputs_retry_successfully() -> None:
    provider = ScriptedProvider(["not json", '{"topic": "camera"}', '{"topic": "support"}'])
    agent = TopicAssignmentAgent(provider, max_assignment_retries=2)

    assert agent.assign_topic("Support replied after three emails.", TOPICS) == "support"
    assert len(provider.calls) == 3


def test_repeated_malformed_output_records_failure_without_local_repair() -> None:
    provider = ScriptedProvider(["not json", '{"topic": "camera"}', '{"label": "design"}'])
    agent = TopicAssignmentAgent(provider, max_assignment_retries=2)

    with pytest.raises(TopicAssignmentError) as excinfo:
        agent.assign_topic("The design is attractive.", TOPICS)

    assert excinfo.value.raw_responses == ("not json", '{"topic": "camera"}', '{"label": "design"}')
    assert len(provider.calls) == 3


def test_parser_requires_json_topic_field_and_allowed_vocabulary() -> None:
    assert _parse_assigned_topic("battery life", TOPICS) is None
    assert _parse_assigned_topic('{"label": "battery life"}', TOPICS) is None
    assert _parse_assigned_topic('{"topic": "camera"}', TOPICS) is None
    assert _parse_assigned_topic('{"topic": "Battery_Life"}', TOPICS) == "battery life"


def test_enrich_table_does_not_read_or_reuse_stored_topic(settings) -> None:
    provider = ScriptedProvider(['{"topics": ["design", "battery life"]}', '{"topic": "battery life"}'])
    agent = TopicAssignmentAgent(provider)
    conn = sqlite3.connect(settings.database_path)
    conn.row_factory = sqlite3.Row
    ensure_review_table(conn, "topic_product")
    insert_review_rows(
        conn,
        "topic_product",
        [
            {
                "title": "Attractive design",
                "content": "Beautiful body but unusable battery life.",
                "language": "en",
                "topic": "design",
            }
        ],
    )

    assert agent.enrich_table(conn, "topic_product") == 1
    row = conn.execute("SELECT topic FROM topic_product").fetchone()

    assert row["topic"] == "battery life"
    assert [call["purpose"] for call in provider.calls] == ["topic_list", "topic_assign"]


def test_paired_language_predictions_are_independent_calls() -> None:
    provider = ScriptedProvider(['{"topic": "battery life"}', '{"topic": "usability"}'])
    agent = TopicAssignmentAgent(provider)

    original = agent.assign_topic("La bateria falla, pero configurar la app tambien fue dificil.", TOPICS)
    translated = agent.assign_topic("The setup was difficult, although the battery also failed.", TOPICS)

    assert original == "battery life"
    assert translated == "usability"
    assert len(provider.calls) == 2


def test_canonical_production_text_avoids_original_and_translation_duplication() -> None:
    english = build_analysis_text(title="Good battery", content="Battery lasts.", translated_review="Good battery Battery lasts.", language="en")
    non_english = build_analysis_text(
        title="Bateria mala",
        content="La bateria falla rapidamente.",
        translated_review="The battery fails quickly.",
        language="es",
    )

    assert english == "Good battery Battery lasts."
    assert non_english == "The battery fails quickly."
    assert "Bateria mala" not in non_english
    assert "La bateria" not in non_english


def test_canonical_production_text_uses_original_title_and_body_when_translation_missing() -> None:
    text = build_analysis_text(
        title="Bateria mala",
        content="La bateria falla rapidamente.",
        translated_review="",
        language="es",
    )

    assert text == "Bateria mala La bateria falla rapidamente."


def test_paired_cross_language_text_rule_uses_body_only_on_both_sides() -> None:
    row = {
        "title": "Titulo original",
        "content": "Contenido original del comentario.",
        "translated_review": "Translated review body.",
    }

    assert _paired_original_text(row) == "Contenido original del comentario."
    assert _paired_translated_text(row) == "Translated review body."
    assert "Titulo" not in _paired_original_text(row)
