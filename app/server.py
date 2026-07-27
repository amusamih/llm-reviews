from __future__ import annotations

import os
from pathlib import Path
import secrets
from typing import Any, Mapping

from dotenv import load_dotenv

from llm_review_analysis.config import ensure_directories, load_settings
from llm_review_analysis.db.connection import connect
from llm_review_analysis.db.schema import list_review_tables
from llm_review_analysis.providers import build_llm_provider
from llm_review_analysis.agents import ConversationState, ReviewOrchestrator


REPO_ROOT = Path(__file__).resolve().parents[1]
SESSION_STATE_KEY = "conversation_state"
PRODUCTION_ENV_VALUES = {"prod", "production", "deploy", "deployed"}


def create_app(*, settings=None, provider=None, orchestrator=None):
    from flask import Flask, jsonify, render_template, request, session

    load_repository_dotenv()
    settings = settings or load_settings()
    ensure_directories(settings)
    provider = provider or build_llm_provider(settings)
    orchestrator = orchestrator or ReviewOrchestrator(settings, provider)

    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.secret_key = resolve_flask_secret_key()
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
    app.config["SESSION_COOKIE_SECURE"] = _secure_cookie_enabled()

    @app.get("/")
    def index():
        return render_template("chat.html")

    @app.post("/chat")
    def chat():
        payload = request.get_json(silent=True) or {}
        prompt = str(payload.get("prompt", "")).strip()
        if not prompt:
            return jsonify({"type": "error", "message": "Prompt is required."}), 400
        with connect(settings.database_path) as conn:
            state_payload = validate_session_state(conn, session.get(SESSION_STATE_KEY))
            result, updated_state, _ = orchestrator.answer_with_state(conn, prompt, state=state_payload)
        serialized_state = serialize_conversation_state(updated_state)
        if _empty_state(serialized_state):
            session.pop(SESSION_STATE_KEY, None)
        else:
            session[SESSION_STATE_KEY] = serialized_state
        return jsonify(result)

    @app.post("/reset")
    def reset():
        session.pop(SESSION_STATE_KEY, None)
        return jsonify({"type": "text", "message": "Conversation context has been reset."})

    return app


def load_repository_dotenv(dotenv_path: str | Path | None = None) -> bool:
    path = Path(dotenv_path) if dotenv_path is not None else REPO_ROOT / ".env"
    return bool(load_dotenv(path, override=False))


def resolve_flask_secret_key(env: Mapping[str, str] | None = None) -> str:
    values = env or os.environ
    configured = str(values.get("FLASK_SECRET_KEY") or "").strip()
    if configured:
        return configured
    if _production_mode(values):
        raise RuntimeError("FLASK_SECRET_KEY must be set for deployed or production Flask use.")
    return secrets.token_hex(32)


def validate_session_state(conn, raw_state: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not raw_state:
        return None
    try:
        state = ConversationState.from_mapping(raw_state)
    except (TypeError, ValueError):
        return None
    if not state.matched_table:
        return state.to_dict()
    if state.matched_table not in set(list_review_tables(conn)):
        return None
    return state.to_dict()


def serialize_conversation_state(state: ConversationState | Mapping[str, Any] | None) -> dict[str, Any]:
    payload = ConversationState.from_mapping(state).to_dict()
    return {
        "product_name": _optional_string(payload.get("product_name")),
        "matched_table": _optional_string(payload.get("matched_table")),
        "active_filters": {str(key): str(value) for key, value in dict(payload.get("active_filters") or {}).items()},
        "previous_route": _optional_string(payload.get("previous_route")),
        "previous_query": _optional_string(payload.get("previous_query")),
        "previous_result_summary": _optional_string(payload.get("previous_result_summary"), limit=360),
        "previous_numeric_result": payload.get("previous_numeric_result")
        if isinstance(payload.get("previous_numeric_result"), (int, float))
        else None,
        "previous_chart_spec": {str(key): str(value) for key, value in dict(payload.get("previous_chart_spec") or {}).items()} or None,
        "evidence_ids": [str(value) for value in list(payload.get("evidence_ids") or [])[:8]],
        "evidence_summary": _optional_string(payload.get("evidence_summary"), limit=360),
        "response_language": _optional_string(payload.get("response_language")),
        "turn_count": int(payload.get("turn_count") or 0),
    }


def _optional_string(value: object, *, limit: int = 240) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text[:limit]


def _empty_state(payload: Mapping[str, Any]) -> bool:
    return payload == ConversationState.reset().to_dict()


def _secure_cookie_enabled(env: Mapping[str, str] | None = None) -> bool:
    values = env or os.environ
    if str(values.get("FLASK_SESSION_COOKIE_SECURE") or "").strip().lower() in {"1", "true", "yes", "on"}:
        return True
    return _production_mode(values)


def _production_mode(env: Mapping[str, str]) -> bool:
    for key in ("FLASK_ENV", "APP_ENV", "ENV"):
        value = str(env.get(key) or "").strip().lower()
        if value in PRODUCTION_ENV_VALUES:
            return True
    return False


if __name__ == "__main__":
    create_app().run(debug=False)
