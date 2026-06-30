from __future__ import annotations

from pathlib import Path
import sqlite3

import pytest

from llm_review_analysis.agents.retrieval_agent import RetrievalAgent
from llm_review_analysis.agents.semantic_reasoning_agent import SemanticReasoningAgent, _load_langchain_semantic_components
from llm_review_analysis.agents.semantic_tagger import SemanticTagger
from llm_review_analysis.db.schema import ensure_review_table, insert_review_rows
from llm_review_analysis.llm import LLMResponse


class RecordingProvider:
    def __init__(self, responses: dict[str, str]) -> None:
        self.responses = responses
        self.calls: list[dict[str, str | None]] = []

    def generate(self, prompt: str, *, purpose: str = "general", response_format: str | None = None) -> LLMResponse:
        self.calls.append({"prompt": prompt, "purpose": purpose, "response_format": response_format})
        return LLMResponse(content=self.responses.get(purpose, ""), model="recording-provider")


class NoopEnrichmentAgent:
    def enrich_table(self, conn: sqlite3.Connection, table_name: str) -> int:
        return 0


class TinyDeterministicEmbeddings:
    vocabulary = ("battery", "charge", "delivery", "broken", "price", "quality")

    def __init__(self, cache_identifier: str = "tiny-deterministic-v1") -> None:
        self.cache_identifier = cache_identifier

    def __call__(self, text: str) -> list[float]:
        return self.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        lower = text.lower()
        vector = [float(lower.count(term)) for term in self.vocabulary]
        return vector or [0.0 for _ in self.vocabulary]


def test_semantic_tagger_uses_provider_backed_path_with_json_output() -> None:
    provider = RecordingProvider(
        {"semantic_tagging": '{"semantic_tags": ["Positive", "Helpful", "No_Justification", "unsupported"]}'}
    )
    tagger = SemanticTagger(provider=provider, use_provider=True)

    tags = tagger.tag_text("Great.")

    assert tags == ["positive", "helpful", "no justification"]
    assert provider.calls == [
        {
            "prompt": provider.calls[0]["prompt"],
            "purpose": "semantic_tagging",
            "response_format": "json",
        }
    ]
    assert "Allowed labels" in str(provider.calls[0]["prompt"])


def test_semantic_tagger_without_provider_uses_offline_fallback() -> None:
    tags = SemanticTagger().tag_text("Great product but bad delivery and not as advertised.")

    assert {"positive", "negative", "contradictory", "potentially misleading"}.issubset(set(tags))


def test_retrieval_enrichment_stores_provider_backed_semantic_tags(settings) -> None:
    provider = RecordingProvider({"semantic_tagging": '{"semantic_tags": ["negative", "helpful"]}'})
    conn = sqlite3.connect(settings.database_path)
    conn.row_factory = sqlite3.Row
    agent = RetrievalAgent(
        settings,
        provider=provider,
        language_agent=NoopEnrichmentAgent(),
        topic_agent=NoopEnrichmentAgent(),
    )

    table = agent.load_records(
        conn,
        "Provider Tagged Product",
        [
            {
                "rating": "1",
                "title": "Stopped charging",
                "content": "Stopped charging after two weeks and support did not help.",
            }
        ],
    )

    row = conn.execute(f"SELECT semantic_tags FROM {table}").fetchone()
    assert row["semantic_tags"] == "negative, helpful"
    assert [call["purpose"] for call in provider.calls] == ["semantic_tagging"]


def test_faiss_semantic_reasoning_retrieves_evidence_with_local_embeddings(settings) -> None:
    provider = RecordingProvider({"semantic_reasoning": "The retrieved review says the battery lasts all day."})
    conn = sqlite3.connect(settings.database_path)
    conn.row_factory = sqlite3.Row
    ensure_review_table(conn, "semantic_product")
    insert_review_rows(
        conn,
        "semantic_product",
        [
            {
                "rating": "5",
                "title": "Battery lasts",
                "content": "Battery lasts all day and charges quickly.",
                "semantic_tags": "positive, helpful",
            },
            {
                "rating": "2",
                "title": "Delivery issue",
                "content": "Delivery was delayed and the packaging was damaged.",
                "semantic_tags": "negative",
            },
        ],
    )
    agent = SemanticReasoningAgent(
        settings=settings,
        provider=provider,
        backend="faiss",
        embedding_model=TinyDeterministicEmbeddings(),
        top_k=2,
    )

    trace = agent.answer_with_trace(conn, "semantic_product", "What do reviews say about battery charge?")

    assert trace.answer == "The retrieved review says the battery lasts all day."
    assert trace.evidence_ids
    assert any("Battery lasts all day" in snippet or "battery lasts all day" in snippet for snippet in trace.evidence_snippets)
    assert provider.calls[-1]["purpose"] == "semantic_reasoning"
    assert "Retrieved review evidence" in str(provider.calls[-1]["prompt"])
    assert "Review ID:" in str(provider.calls[-1]["prompt"])


def test_faiss_first_call_writes_safe_cache_and_second_call_reuses_it(settings, monkeypatch) -> None:
    pytest.importorskip("faiss")
    components = _load_langchain_semantic_components()
    provider = RecordingProvider({"semantic_reasoning": "Cached evidence was used."})
    conn = _semantic_cache_connection(settings)
    agent = SemanticReasoningAgent(
        settings=settings,
        provider=provider,
        backend="faiss",
        embedding_model=TinyDeterministicEmbeddings(),
        langchain_components=components,
    )

    first_trace = agent.answer_with_trace(conn, "semantic_cache_product", "What mentions battery charge?")

    assert first_trace.evidence_ids
    cache_files = _cache_files(settings.vectorstore_dir)
    assert any(path.name == "manifest.json" for path in cache_files)
    assert any(path.name == "documents.json" for path in cache_files)
    assert any(path.name == "index.faiss" for path in cache_files)

    def fail_rebuild(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("FAISS cache was not reused.")

    monkeypatch.setattr(components["FAISS"], "from_documents", fail_rebuild)
    second_agent = SemanticReasoningAgent(
        settings=settings,
        provider=provider,
        backend="faiss",
        embedding_model=TinyDeterministicEmbeddings(),
        langchain_components=components,
    )

    second_trace = second_agent.answer_with_trace(conn, "semantic_cache_product", "What mentions battery charge?")

    assert second_trace.evidence_ids
    assert "Cached evidence was used." in second_trace.answer


def test_faiss_cache_invalidates_when_review_content_changes(settings, monkeypatch) -> None:
    pytest.importorskip("faiss")
    components = _load_langchain_semantic_components()
    provider = RecordingProvider({"semantic_reasoning": "Rebuilt after content change."})
    conn = _semantic_cache_connection(settings)
    agent = SemanticReasoningAgent(
        settings=settings,
        provider=provider,
        backend="faiss",
        embedding_model=TinyDeterministicEmbeddings(),
        langchain_components=components,
    )
    agent.answer_with_trace(conn, "semantic_cache_product", "What mentions battery charge?")
    original_from_documents = components["FAISS"].from_documents
    rebuild_count = 0

    def counted_from_documents(*args, **kwargs):  # noqa: ANN002, ANN003
        nonlocal rebuild_count
        rebuild_count += 1
        return original_from_documents(*args, **kwargs)

    conn.execute(
        "UPDATE semantic_cache_product SET content = ? WHERE id = 1",
        ("Battery charge now lasts two days after the update.",),
    )
    conn.commit()
    monkeypatch.setattr(components["FAISS"], "from_documents", counted_from_documents)
    second_agent = SemanticReasoningAgent(
        settings=settings,
        provider=provider,
        backend="faiss",
        embedding_model=TinyDeterministicEmbeddings(),
        langchain_components=components,
    )

    second_agent.answer_with_trace(conn, "semantic_cache_product", "What mentions battery charge?")

    assert rebuild_count == 1
    assert len(list((settings.vectorstore_dir / "semantic_cache_product").iterdir())) >= 2


def test_faiss_cache_invalidates_when_embedding_configuration_changes(settings, monkeypatch) -> None:
    pytest.importorskip("faiss")
    components = _load_langchain_semantic_components()
    provider = RecordingProvider({"semantic_reasoning": "Rebuilt after embedding change."})
    conn = _semantic_cache_connection(settings)
    first_agent = SemanticReasoningAgent(
        settings=settings,
        provider=provider,
        backend="faiss",
        embedding_model=TinyDeterministicEmbeddings(cache_identifier="tiny-v1"),
        langchain_components=components,
    )
    first_agent.answer_with_trace(conn, "semantic_cache_product", "What mentions battery charge?")
    original_from_documents = components["FAISS"].from_documents
    rebuild_count = 0

    def counted_from_documents(*args, **kwargs):  # noqa: ANN002, ANN003
        nonlocal rebuild_count
        rebuild_count += 1
        return original_from_documents(*args, **kwargs)

    monkeypatch.setattr(components["FAISS"], "from_documents", counted_from_documents)
    second_agent = SemanticReasoningAgent(
        settings=settings,
        provider=provider,
        backend="faiss",
        embedding_model=TinyDeterministicEmbeddings(cache_identifier="tiny-v2"),
        langchain_components=components,
    )

    second_agent.answer_with_trace(conn, "semantic_cache_product", "What mentions battery charge?")

    assert rebuild_count == 1
    assert len(list((settings.vectorstore_dir / "semantic_cache_product").iterdir())) >= 2


def test_faiss_corrupt_or_incomplete_cache_is_rebuilt_safely(settings, monkeypatch) -> None:
    pytest.importorskip("faiss")
    components = _load_langchain_semantic_components()
    provider = RecordingProvider({"semantic_reasoning": "Rebuilt after corrupt cache."})
    conn = _semantic_cache_connection(settings)
    agent = SemanticReasoningAgent(
        settings=settings,
        provider=provider,
        backend="faiss",
        embedding_model=TinyDeterministicEmbeddings(),
        langchain_components=components,
    )
    agent.answer_with_trace(conn, "semantic_cache_product", "What mentions battery charge?")
    for documents_path in settings.vectorstore_dir.rglob("documents.json"):
        documents_path.write_text("{not valid json", encoding="utf-8")
        break
    original_from_documents = components["FAISS"].from_documents
    rebuild_count = 0

    def counted_from_documents(*args, **kwargs):  # noqa: ANN002, ANN003
        nonlocal rebuild_count
        rebuild_count += 1
        return original_from_documents(*args, **kwargs)

    monkeypatch.setattr(components["FAISS"], "from_documents", counted_from_documents)
    rebuilt_trace = agent.answer_with_trace(conn, "semantic_cache_product", "What mentions battery charge?")

    assert rebuild_count == 1
    assert rebuilt_trace.evidence_ids


def test_faiss_cache_uses_no_unsafe_deserialization_path() -> None:
    source = Path("src/llm_review_analysis/agents/semantic_reasoning_agent.py").read_text(encoding="utf-8")

    assert "pickle.load" not in source
    assert "allow_dangerous_deserialization" not in source
    assert ".load_local" not in source
    assert "eval(" not in source
    assert "exec(" not in source


def test_faiss_semantic_reasoning_returns_controlled_response_when_no_evidence(settings) -> None:
    provider = RecordingProvider({"semantic_reasoning": "This should not be called."})
    conn = sqlite3.connect(settings.database_path)
    conn.row_factory = sqlite3.Row
    ensure_review_table(conn, "empty_semantic_product")
    agent = SemanticReasoningAgent(
        settings=settings,
        provider=provider,
        backend="faiss",
        embedding_model=TinyDeterministicEmbeddings(),
    )

    trace = agent.answer_with_trace(conn, "empty_semantic_product", "Why is it good?")

    assert trace.answer == "No matching reviews were available for semantic reasoning."
    assert trace.evidence_ids == ()
    assert trace.evidence_snippets == ()
    assert provider.calls == []


def _semantic_cache_connection(settings) -> sqlite3.Connection:
    conn = sqlite3.connect(settings.database_path)
    conn.row_factory = sqlite3.Row
    ensure_review_table(conn, "semantic_cache_product")
    insert_review_rows(
        conn,
        "semantic_cache_product",
        [
            {
                "rating": "5",
                "title": "Battery charge",
                "content": "Battery charge lasts all day and the quality is strong.",
                "semantic_tags": "positive, helpful",
            },
            {
                "rating": "1",
                "title": "Broken delivery",
                "content": "Delivery was broken and the packaging quality was bad.",
                "semantic_tags": "negative",
            },
        ],
    )
    return conn


def _cache_files(vectorstore_dir: Path) -> list[Path]:
    return [path for path in vectorstore_dir.rglob("*") if path.is_file()]
