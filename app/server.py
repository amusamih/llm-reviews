from __future__ import annotations

import os
from pathlib import Path
import secrets
from typing import Any, Mapping

from dotenv import load_dotenv

from llm_review_analysis.config import ensure_directories, load_settings
from llm_review_analysis.db.connection import connect
from llm_review_analysis.db.schema import list_review_tables, validate_identifier
from llm_review_analysis.providers import build_llm_provider
from llm_review_analysis.agents import ConversationState, ReviewOrchestrator


REPO_ROOT = Path(__file__).resolve().parents[1]
SESSION_STATE_KEY = "conversation_state"
PRODUCTION_ENV_VALUES = {"prod", "production", "deploy", "deployed"}
PLACEHOLDER_SECRET_VALUES = {
    "replace-with-a-long-random-secret",
    "replace-with-a-random-secret",
    "change-me",
    "changeme",
    "secret",
    "development-secret",
    "dev-secret",
    "your-secret-key",
}
MAX_SESSION_FILTERS = 6
MAX_FILTER_KEY_LENGTH = 40
MAX_FILTER_VALUE_LENGTH = 80
MAX_EVIDENCE_IDS = 8
MAX_EVIDENCE_ID_LENGTH = 48
MAX_CHART_SPEC_FIELDS = 10
MAX_CHART_SPEC_KEY_LENGTH = 40
MAX_CHART_SPEC_VALUE_LENGTH = 80
MAX_SUMMARY_LENGTH = 360
MAX_PRODUCT_LENGTH = 120
MAX_TABLE_LENGTH = 80
MAX_ROUTE_LENGTH = 32
MAX_QUERY_LENGTH = 320
MAX_LANGUAGE_LENGTH = 24
MAX_TURN_COUNT = 99


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
    if configured and configured.lower() not in PLACEHOLDER_SECRET_VALUES:
        return configured
    if _production_mode(values):
        raise RuntimeError("FLASK_SECRET_KEY must be set to a non-placeholder value for deployed or production Flask use.")
    return secrets.token_urlsafe(48)


def validate_session_state(conn, raw_state: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not raw_state or not isinstance(raw_state, Mapping):
        return None
    serialized = serialize_conversation_state(raw_state)
    try:
        state = ConversationState.from_mapping(serialized)
    except (TypeError, ValueError):
        return None
    if not state.matched_table:
        return serialized
    try:
        table = validate_identifier(state.matched_table)
    except ValueError:
        return None
    if table not in set(list_review_tables(conn)):
        return None
    return serialized


def serialize_conversation_state(state: ConversationState | Mapping[str, Any] | None) -> dict[str, Any]:
    if state is None:
        payload: Mapping[str, Any] = ConversationState.reset().to_dict()
    elif isinstance(state, ConversationState):
        payload = state.to_dict()
    elif isinstance(state, Mapping):
        payload = state
    else:
        payload = ConversationState.reset().to_dict()
    return {
        "product_name": _optional_string(payload.get("product_name"), limit=MAX_PRODUCT_LENGTH),
        "matched_table": _optional_string(payload.get("matched_table"), limit=MAX_TABLE_LENGTH),
        "active_filters": _simple_string_mapping(
            payload.get("active_filters"),
            max_items=MAX_SESSION_FILTERS,
            key_limit=MAX_FILTER_KEY_LENGTH,
            value_limit=MAX_FILTER_VALUE_LENGTH,
        ),
        "previous_route": _optional_string(payload.get("previous_route"), limit=MAX_ROUTE_LENGTH),
        "previous_query": _optional_string(payload.get("previous_query"), limit=MAX_QUERY_LENGTH),
        "previous_result_summary": _optional_string(payload.get("previous_result_summary"), limit=MAX_SUMMARY_LENGTH),
        "previous_numeric_result": _safe_numeric(payload.get("previous_numeric_result")),
        "previous_chart_spec": _simple_string_mapping(
            payload.get("previous_chart_spec"),
            max_items=MAX_CHART_SPEC_FIELDS,
            key_limit=MAX_CHART_SPEC_KEY_LENGTH,
            value_limit=MAX_CHART_SPEC_VALUE_LENGTH,
        )
        or None,
        "evidence_ids": _simple_string_sequence(
            payload.get("evidence_ids"),
            max_items=MAX_EVIDENCE_IDS,
            value_limit=MAX_EVIDENCE_ID_LENGTH,
        ),
        "evidence_summary": _optional_string(payload.get("evidence_summary"), limit=MAX_SUMMARY_LENGTH),
        "response_language": _optional_string(payload.get("response_language"), limit=MAX_LANGUAGE_LENGTH),
        "turn_count": _safe_turn_count(payload.get("turn_count")),
    }


def _optional_string(value: object, *, limit: int = 240) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text[:limit]


def _simple_string_mapping(value: object, *, max_items: int, key_limit: int, value_limit: int) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, str] = {}
    for key, raw_value in value.items():
        if len(result) >= max_items:
            break
        if _is_nested_value(raw_value):
            continue
        clean_key = _optional_string(key, limit=key_limit)
        clean_value = _optional_string(raw_value, limit=value_limit)
        if clean_key and clean_value is not None:
            result[clean_key] = clean_value
    return result


def _simple_string_sequence(value: object, *, max_items: int, value_limit: int) -> list[str]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, (list, tuple)):
        return []
    result: list[str] = []
    for item in value:
        if len(result) >= max_items:
            break
        if _is_nested_value(item):
            continue
        clean_item = _optional_string(item, limit=value_limit)
        if clean_item is not None:
            result.append(clean_item)
    return result


def _is_nested_value(value: object) -> bool:
    return isinstance(value, (Mapping, list, tuple, set))


def _safe_numeric(value: object) -> int | float | None:
    if isinstance(value, bool):
        return None
    return value if isinstance(value, (int, float)) else None


def _safe_turn_count(value: object) -> int:
    try:
        count = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return max(0, min(count, MAX_TURN_COUNT))


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
