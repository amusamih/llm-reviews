from __future__ import annotations

import json
import sqlite3
from typing import Any, Mapping

from app.server import SESSION_STATE_KEY, create_app
from llm_review_analysis.agents import ConversationState, ReviewOrchestrator


def test_flask_chat_creates_compact_conversation_state(settings):
    orchestrator = FakeStatefulOrchestrator()
    app = create_app(settings=settings, provider=object(), orchestrator=orchestrator)
    client = app.test_client()

    response = client.post("/chat", json={"prompt": "How many reviews for sample product?"})

    assert response.status_code == 200
    with client.session_transaction() as session:
        state = session[SESSION_STATE_KEY]
    assert state["product_name"] == "sample product"
    assert state["matched_table"] == "sample_product"
    assert state["previous_route"] == "DIRECT_SQL"
    json.dumps(state)


def test_flask_followup_uses_inherited_product_context(settings):
    orchestrator = FakeStatefulOrchestrator()
    app = create_app(settings=settings, provider=object(), orchestrator=orchestrator)
    client = app.test_client()

    client.post("/chat", json={"prompt": "How many reviews for sample product?"})
    response = client.post("/chat", json={"prompt": "Now do the same for five-star reviews."})

    assert response.status_code == 200
    assert response.get_json()["message"] == "sample_product rating 5"
    assert orchestrator.calls[-1]["state"]["matched_table"] == "sample_product"
    with client.session_transaction() as session:
        assert session[SESSION_STATE_KEY]["active_filters"] == {"rating": "5"}


def test_flask_product_switch_replaces_session_context(settings):
    orchestrator = FakeStatefulOrchestrator()
    app = create_app(settings=settings, provider=object(), orchestrator=orchestrator)
    client = app.test_client()

    client.post("/chat", json={"prompt": "How many reviews for sample product?"})
    client.post("/chat", json={"prompt": "How many reviews for other product?"})

    with client.session_transaction() as session:
        state = session[SESSION_STATE_KEY]
    assert state["product_name"] == "other product"
    assert state["matched_table"] == "other_product"
    assert state["active_filters"] == {}


def test_flask_reset_route_clears_only_current_session(settings):
    app = create_app(settings=settings, provider=object(), orchestrator=FakeStatefulOrchestrator())
    first = app.test_client()
    second = app.test_client()

    first.post("/chat", json={"prompt": "How many reviews for sample product?"})
    second.post("/chat", json={"prompt": "How many reviews for other product?"})
    response = first.post("/reset")

    assert response.status_code == 200
    with first.session_transaction() as session:
        assert SESSION_STATE_KEY not in session
    with second.session_transaction() as session:
        assert session[SESSION_STATE_KEY]["matched_table"] == "other_product"


def test_flask_reset_prompt_clears_session(settings):
    app = create_app(settings=settings, provider=object(), orchestrator=FakeStatefulOrchestrator())
    client = app.test_client()

    client.post("/chat", json={"prompt": "How many reviews for sample product?"})
    response = client.post("/chat", json={"prompt": "reset context"})

    assert response.status_code == 200
    with client.session_transaction() as session:
        assert SESSION_STATE_KEY not in session


def test_flask_session_does_not_store_large_response_payloads(settings):
    app = create_app(settings=settings, provider=object(), orchestrator=FakeStatefulOrchestrator())
    client = app.test_client()

    response = client.post("/chat", json={"prompt": "Show chart for sample product."})

    assert response.status_code == 200
    assert "chart_b64" in response.get_json()
    with client.session_transaction() as session:
        state = session[SESSION_STATE_KEY]
    serialized = json.dumps(state)
    assert "chart_b64" not in serialized
    assert len(serialized) < 2000


def test_programmatic_stateless_answer_path_remains_available():
    assert callable(ReviewOrchestrator.answer)


class FakeStatefulOrchestrator:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def answer_with_state(
        self,
        conn: sqlite3.Connection,
        prompt: str,
        *,
        state: Mapping[str, Any] | None = None,
        product_table: str | None = None,
        reset: bool = False,
    ):
        del conn, product_table, reset
        previous = ConversationState.from_mapping(state)
        self.calls.append({"prompt": prompt, "state": previous.to_dict()})
        if prompt.lower() in {"reset", "reset context", "clear context"}:
            return {"type": "text", "message": "Conversation context has been reset."}, ConversationState.reset(), {}

        product_name = "other product" if "other product" in prompt.lower() else previous.product_name or "sample product"
        table = product_name.replace(" ", "_")
        filters = {"rating": "5"} if "five-star" in prompt.lower() else {}
        route = "ANALYTICS" if "chart" in prompt.lower() else "DIRECT_SQL"
        result = {
            "type": "chart",
            "message": f"{table} chart",
            "chart_b64": "x" * 5000,
        } if route == "ANALYTICS" else {
            "type": "text",
            "message": f"{table} rating {filters.get('rating', 'all')}",
        }
        state_out = ConversationState(
            product_name=product_name,
            matched_table=table,
            active_filters=filters,
            previous_route=route,
            previous_query=prompt,
            previous_result_summary=result["message"],
            previous_chart_spec={"chart_type": "bar", "group_by": "rating", "aggregation": "count"} if route == "ANALYTICS" else None,
            evidence_ids=("1", "2"),
            evidence_summary="compact evidence",
            response_language="en",
            turn_count=min(previous.turn_count + 1, 9),
        )
        return result, state_out, {}
