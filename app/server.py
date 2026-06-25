from __future__ import annotations

from llm_review_analysis.config import ensure_directories, load_settings
from llm_review_analysis.db.connection import connect
from llm_review_analysis.providers import build_llm_provider
from llm_review_analysis.agents import ReviewOrchestrator


def create_app():
    from flask import Flask, jsonify, render_template, request

    settings = load_settings()
    ensure_directories(settings)
    provider = build_llm_provider(settings)
    orchestrator = ReviewOrchestrator(settings, provider)

    app = Flask(__name__, template_folder="templates", static_folder="static")

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
            result = orchestrator.answer(conn, prompt)
        return jsonify(result)

    return app


if __name__ == "__main__":
    create_app().run(debug=False)
