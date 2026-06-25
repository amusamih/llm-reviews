from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from typing import Any

from llm_review_analysis.config import Settings
from llm_review_analysis.db.schema import REVIEW_COLUMNS, validate_identifier
from llm_review_analysis.db.sql_validator import execute_validated_select
from llm_review_analysis.llm import LLMProvider


@dataclass(frozen=True)
class SemanticReasoningTrace:
    answer: str
    evidence_ids: tuple[str, ...]
    evidence_snippets: tuple[str, ...]


class SemanticReasoningAgent:
    """Retrieval-backed interpretive response path.

    The default implementation is deterministic and fixture-friendly. The
    paper-consistent live path uses LangChain + FAISS and remains explicitly
    gated because it needs live embeddings/LLM calls.
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        provider: LLMProvider | None = None,
        backend: str = "lexical",
        top_k: int = 6,
    ) -> None:
        self.settings = settings
        self.provider = provider
        self.backend = backend
        self.top_k = top_k

    def answer(self, conn: sqlite3.Connection, table_name: str, prompt: str) -> str:
        return self.answer_with_trace(conn, table_name, prompt).answer

    def answer_with_trace(self, conn: sqlite3.Connection, table_name: str, prompt: str) -> SemanticReasoningTrace:
        if self.backend == "faiss":
            return self._answer_with_langchain_faiss(conn, table_name, prompt)
        table = validate_identifier(table_name)
        columns, rows = _fetch_reasoning_rows(conn, table)
        if not rows:
            return SemanticReasoningTrace(
                answer="No matching reviews were available for semantic reasoning.",
                evidence_ids=(),
                evidence_snippets=(),
            )
        ranked = _rank_rows(prompt, columns, rows)[:5]
        if not ranked:
            return SemanticReasoningTrace(
                answer="No semantically relevant review snippets were found.",
                evidence_ids=(),
                evidence_snippets=(),
            )
        evidence_ids = tuple(review_id for review_id, _ in ranked)
        snippets = tuple(text for _, text in ranked)
        return SemanticReasoningTrace(
            answer="Relevant review evidence: " + " | ".join(snippets),
            evidence_ids=evidence_ids,
            evidence_snippets=snippets,
        )

    def _answer_with_langchain_faiss(self, conn: sqlite3.Connection, table_name: str, prompt: str) -> SemanticReasoningTrace:
        if self.settings is None or self.provider is None:
            raise RuntimeError("FAISS semantic reasoning requires settings and a live LLM provider.")
        if not self.settings.allow_live_llm:
            raise RuntimeError("FAISS semantic reasoning requires ALLOW_LIVE_LLM=true because embeddings and reasoning use live model calls.")

        components = _load_langchain_semantic_components()
        table = validate_identifier(table_name)
        columns, rows = _fetch_reasoning_rows(conn, table)
        documents = _rows_to_documents(columns, rows)
        if not documents:
            return SemanticReasoningTrace(
                answer="No matching reviews were available for semantic reasoning.",
                evidence_ids=(),
                evidence_snippets=(),
            )

        splitter = components["RecursiveCharacterTextSplitter"](chunk_size=1000, chunk_overlap=120)
        chunks = splitter.split_documents(documents)
        embeddings = components["OpenAIEmbeddings"](model=self.settings.embedding_model)
        vectorstore = components["FAISS"].from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": self.top_k})
        vector_retrieved = _invoke_retriever(retriever, prompt)
        lexical_retrieved = _lexically_matching_documents(prompt, documents, limit=self.top_k)
        retrieved = _merge_documents([*lexical_retrieved, *vector_retrieved], limit=self.top_k)
        context = _format_retrieved_context(prompt, retrieved)
        if not context.strip():
            return SemanticReasoningTrace(
                answer="No semantically relevant review snippets were found.",
                evidence_ids=(),
                evidence_snippets=(),
            )

        response = self.provider.generate(_reasoning_prompt(prompt, context), purpose="semantic_reasoning")
        evidence_ids = tuple(
            str(getattr(doc, "metadata", {}).get("review_id"))
            for doc in retrieved
            if getattr(doc, "metadata", {}).get("review_id") is not None
        )
        evidence_snippets = tuple(getattr(doc, "page_content", str(doc)) for doc in retrieved)
        return SemanticReasoningTrace(
            answer=response.content.strip() or "No semantic reasoning response was generated.",
            evidence_ids=evidence_ids,
            evidence_snippets=evidence_snippets,
        )


def _fetch_reasoning_rows(conn: sqlite3.Connection, table: str) -> tuple[list[str], list[tuple]]:
    sql = f"SELECT id, rating, title, content, translated_review, semantic_tags FROM {table}"
    return execute_validated_select(conn, sql, allowed_tables=[table], allowed_columns=REVIEW_COLUMNS)


def _rows_to_documents(columns: list[str], rows: list[tuple]) -> list[Any]:
    Document = _load_document_class()
    documents = []
    for row in rows:
        row_map = dict(zip(columns, row))
        content = " ".join(
            str(row_map.get(col) or "") for col in ("title", "content", "translated_review", "semantic_tags")
        ).strip()
        if not content:
            continue
        metadata = {
            "review_id": row_map.get("id"),
            "rating": row_map.get("rating"),
            "semantic_tags": row_map.get("semantic_tags"),
        }
        documents.append(Document(page_content=content, metadata=metadata))
    return documents


def _rank_rows(prompt: str, columns: list[str], rows: list[tuple]) -> list[tuple[str, str]]:
    prompt_terms = _salient_terms(prompt)
    phrases = _query_phrases(prompt)
    scored: list[tuple[int, str, str]] = []
    for row in rows:
        row_map = dict(zip(columns, row))
        text = " ".join(str(row_map.get(col) or "") for col in ("title", "content", "translated_review", "semantic_tags"))
        lowered_text = text.lower()
        terms = set(re.findall(r"[A-Za-z0-9]+", lowered_text))
        score = len(prompt_terms.intersection(terms))
        if any(phrase in lowered_text for phrase in phrases):
            score += 100
        scored.append((score, str(row_map.get("id", "")), text.strip()))
    scored.sort(key=lambda item: item[0], reverse=True)
    positives = [(review_id, text) for score, review_id, text in scored if score > 0 and text]
    if positives:
        return positives
    return [(review_id, text) for _, review_id, text in scored if text]


SEMANTIC_RETRIEVAL_STOPWORDS = {
    "about",
    "amazon",
    "because",
    "does",
    "for",
    "from",
    "have",
    "mention",
    "mentions",
    "product",
    "products",
    "review",
    "reviews",
    "that",
    "the",
    "this",
    "users",
    "what",
    "which",
    "with",
}


def _lexically_matching_documents(prompt: str, documents: list[Any], *, limit: int) -> list[Any]:
    terms = _salient_terms(prompt)
    phrases = _query_phrases(prompt)
    scored: list[tuple[int, int, Any]] = []
    for index, doc in enumerate(documents):
        content = str(getattr(doc, "page_content", doc))
        lowered = content.lower()
        doc_terms = set(re.findall(r"[A-Za-z0-9]+", lowered))
        score = len(terms.intersection(doc_terms))
        if any(phrase in lowered for phrase in phrases):
            score += 100
        if score > 0:
            scored.append((score, index, doc))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [doc for _, _, doc in scored[:limit]]


def _merge_documents(documents: list[Any], *, limit: int) -> list[Any]:
    merged: list[Any] = []
    seen: set[tuple[str, str]] = set()
    for doc in documents:
        metadata = getattr(doc, "metadata", {}) if hasattr(doc, "metadata") else {}
        review_id = str(metadata.get("review_id", "")) if isinstance(metadata, dict) else ""
        content = str(getattr(doc, "page_content", doc))
        key = (review_id, content)
        if key in seen:
            continue
        seen.add(key)
        merged.append(doc)
        if len(merged) >= limit:
            break
    return merged


def _salient_terms(prompt: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z0-9]+", prompt.lower())
        if len(token) > 2 and token not in SEMANTIC_RETRIEVAL_STOPWORDS
    }


def _query_phrases(prompt: str) -> tuple[str, ...]:
    phrases: list[str] = []
    for pattern in (
        r"\bmention\s+(.+?)\s+for\b",
        r"\bmentions\s+(.+?)\s+for\b",
        r"['\"]([^'\"]{4,})['\"]",
    ):
        for match in re.finditer(pattern, prompt, flags=re.IGNORECASE):
            phrase = " ".join(match.group(1).lower().split()).strip(" ?.!,:;\"'")
            if phrase and phrase not in phrases:
                phrases.append(phrase)
    return tuple(phrases)


def _load_langchain_semantic_components() -> dict[str, Any]:
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_openai import OpenAIEmbeddings
    except ImportError as exc:
        raise RuntimeError(
            "FAISS semantic reasoning requires LangChain paper-stack dependencies. "
            "Install with: python -m pip install -e \".[paper]\""
        ) from exc

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
        except ImportError as exc:
            raise RuntimeError("LangChain text splitter dependency is unavailable.") from exc

    return {
        "FAISS": FAISS,
        "OpenAIEmbeddings": OpenAIEmbeddings,
        "RecursiveCharacterTextSplitter": RecursiveCharacterTextSplitter,
    }


def _load_document_class() -> Any:
    try:
        from langchain_core.documents import Document

        return Document
    except ImportError:
        try:
            from langchain.schema import Document

            return Document
        except ImportError as exc:
            raise RuntimeError(
                "LangChain Document class is unavailable. Install with: python -m pip install -e \".[paper]\""
            ) from exc


def _invoke_retriever(retriever: Any, prompt: str) -> list[Any]:
    if hasattr(retriever, "invoke"):
        return list(retriever.invoke(prompt))
    return list(retriever.get_relevant_documents(prompt))


def _format_retrieved_context(prompt: str, documents: list[Any]) -> str:
    return "\n\n".join(_document_context_entry(prompt, doc) for doc in documents)


def _document_context_entry(prompt: str, doc: Any) -> str:
    metadata = getattr(doc, "metadata", {}) if hasattr(doc, "metadata") else {}
    review_id = metadata.get("review_id") if isinstance(metadata, dict) else None
    content = _focused_excerpt(prompt, str(getattr(doc, "page_content", doc)))
    if review_id is None:
        return f"[Review ID: unavailable]\n{content}"
    return f"[Review ID: {review_id}]\n{content}"


def _focused_excerpt(prompt: str, content: str, *, window: int = 450) -> str:
    compact = _normalize_whitespace(content)
    lowered = compact.lower()
    phrases = _query_phrases(prompt)
    for phrase in phrases:
        index = lowered.find(phrase)
        if index >= 0:
            start = max(0, index - window)
            end = min(len(compact), index + len(phrase) + window)
            prefix = "... " if start > 0 else ""
            suffix = " ..." if end < len(compact) else ""
            return f"{prefix}{compact[start:end]}{suffix}"
    return compact[:1200] + (" ..." if len(compact) > 1200 else "")


def _normalize_whitespace(text: str) -> str:
    return " ".join(str(text).split())


def _reasoning_prompt(prompt: str, context: str) -> str:
    return (
        "Answer the user question using only the retrieved review evidence. "
        "Treat product/table names in the question as the review corpus being queried; "
        "do not require the review text itself to mention that corpus name. "
        "If the question asks about an exact phrase and that phrase appears in the evidence, "
        "quote or closely paraphrase the phrase and explain what the customer was discussing. "
        "Cite the relevant review ID(s). "
        "If the evidence is insufficient, say so clearly.\n\n"
        f"User question:\n{prompt}\n\n"
        f"Retrieved review evidence:\n{context}"
    )

