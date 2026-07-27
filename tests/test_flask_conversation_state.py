from __future__ import annotations

import json
import sqlite3
from typing import Any, Mapping

import pytest

from app.server import (
    SESSION_STATE_KEY,
    create_app,
    load_repository_dotenv,
    resolve_flask_secret_key,
)
from llm_review_analysis.agents import ConversationState, ReviewOrchestrator
from llm_review_analysis.db.schema import ensure_review_table


def test_repository_root_dotenv_values_load_without_overwriting_process_env(tmp_path, monkeypatch):
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("FLASK_SECRET_KEY=from-dotenv\nLLM_PROVIDER=langchain\n", encoding="utf-8")
    monkeypatch.delenv("FLASK_SECRET_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "mock")

    assert load_repository_dotenv(dotenv_path) is True

    assert resolve_flask_secret_key() == "from-dotenv"
    assert __import__("os").environ["LLM_PROVIDER"] == "mock"


def test_configured_flask_secret_key_is_used():
    assert resolve_flask_secret_key({"FLASK_SECRET_KEY": "configured-secret"}) == "configured-secret"


def test_local_development_secret_is_ephemeral_and_not_former_fixed_value():
    first = resolve_flask_secret_key({"APP_ENV": "development"})
    second = resolve_flask_secret_key({"APP_ENV": "development"})

    assert first
    assert second
    assert first != second
    assert first != "llm-review-analysis-dev-secret"


def test_missing_flask_secret_fails_in_production_mode():
    with pytest.raises(RuntimeError, match="FLASK_SECRET_KEY"):
        resolve_flask_secret_key({"APP_ENV": "production"})


def test_flask_cookie_settings_are_directly_configured(settings, monkeypatch):
    monkeypatch.setenv("FLASK_SECRET_KEY", "unit-secret")
    app = create_app(settings=settings, provider=object(), orchestrator=FakeStatefulOrchestrator())

    assert app.config["SESSION_COOKIE_HTTPONLY"] is True
    assert app.config["SESSION_COOKIE_SAMESITE"] == "Lax"
    assert app.config["SESSION_COOKIE_SECURE"] is False


def test_flask_chat_creates_compact_conversation_state(settings, monkeypatch):
    monkeypatch.setenv("FLASK_SECRET_KEY", "unit-secret")
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


def test_flask_followup_uses_inherited_product_context(settings, monkeypatch):
    monkeypatch.setenv("FLASK_SECRET_KEY", "unit-secret")
    _ensure_tables(settings, "sample_product")
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


def test_flask_product_switch_replaces_session_context(settings, monkeypatch):
    monkeypatch.setenv("FLASK_SECRET_KEY", "unit-secret")
    _ensure_tables(settings, "sample_product", "other_product")
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


def test_flask_reset_route_clears_only_current_session(settings, monkeypatch):
    monkeypatch.setenv("FLASK_SECRET_KEY", "unit-secret")
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


def test_flask_reset_prompt_clears_session(settings, monkeypatch):
    monkeypatch.setenv("FLASK_SECRET_KEY", "unit-secret")
    _ensure_tables(settings, "sample_product")
    app = create_app(settings=settings, provider=object(), orchestrator=FakeStatefulOrchestrator())
    client = app.test_client()

    client.post("/chat", json={"prompt": "How many reviews for sample product?"})
    response = client.post("/chat", json={"prompt": "reset context"})

    assert response.status_code == 200
    with client.session_transaction() as session:
        assert SESSION_STATE_KEY not in session


def test_flask_session_does_not_store_large_response_payloads(settings, monkeypatch):
    monkeypatch.setenv("FLASK_SECRET_KEY", "unit-secret")
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


def test_flask_drops_forged_or_stale_matched_table_from_session(settings, monkeypatch):
    monkeypatch.setenv("FLASK_SECRET_KEY", "unit-secret")
    orchestrator = FakeStatefulOrchestrator()
    app = create_app(settings=settings, provider=object(), orchestrator=orchestrator)
    client = app.test_client()
    with client.session_transaction() as session:
        session[SESSION_STATE_KEY] = ConversationState(product_name="forged product", matched_table="forged_table").to_dict()

    response = client.post("/chat", json={"prompt": "Now do the same for five-star reviews."})

    assert response.status_code == 200
    assert orchestrator.calls[-1]["state"]["matched_table"] is None
    assert orchestrator.calls[-1]["state"]["product_name"] is None


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


def _ensure_tables(settings, *tables: str) -> None:
    conn = sqlite3.connect(settings.database_path)
    try:
        for table in tables:
            ensure_review_table(conn, table)
    finally:
        conn.close()
