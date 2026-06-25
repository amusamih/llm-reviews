from pathlib import Path

from llm_review_analysis.config import load_settings


def test_default_settings_are_mock_and_project_relative():
    settings = load_settings({})
    assert settings.llm_provider == "mock"
    assert settings.semantic_retrieval_backend == "lexical"
    assert not settings.allow_live_llm
    assert not settings.allow_live_retrieval


def test_langchain_live_settings_default_to_faiss_semantic_backend():
    settings = load_settings({"LLM_PROVIDER": "langchain", "ALLOW_LIVE_LLM": "true"})
    assert settings.llm_provider == "langchain"
    assert settings.semantic_retrieval_backend == "faiss"


def test_new_implementation_has_no_private_hp_path():
    text = "\n".join(path.read_text(encoding="utf-8") for path in Path("src/llm_review_analysis").rglob("*.py"))
    assert "C:/Users/hp" not in text
    assert "C:\\Users\\hp" not in text
