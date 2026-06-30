from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any, Iterable, Mapping

from llm_review_analysis.config import Settings
from llm_review_analysis.db.schema import ensure_review_table, insert_review_rows, normalize_table_name
from llm_review_analysis.llm import LLMProvider, MockLLMProvider

from .language_agent import LanguageAgent
from .semantic_tagger import SemanticTagger
from .topic_assignment_agent import TopicAssignmentAgent


class RetrievalError(RuntimeError):
    pass


class RetrievalAgent:
    """Offline-first retrieval/persistence agent.

    Live scraping is intentionally disabled here. Live retrieval experiments
    should use fixture data or an explicitly approved retrieval adapter.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        provider: LLMProvider | None = None,
        language_agent: Any | None = None,
        topic_agent: Any | None = None,
        semantic_tagger: Any | None = None,
        enrich_after_load: bool = True,
    ) -> None:
        self.settings = settings
        enrichment_provider = provider or MockLLMProvider()
        self.language_agent = language_agent or LanguageAgent(enrichment_provider)
        self.topic_agent = topic_agent or TopicAssignmentAgent(enrichment_provider)
        self.semantic_tagger = semantic_tagger or SemanticTagger(
            provider=enrichment_provider,
            use_provider=provider is not None,
        )
        self.enrich_after_load = enrich_after_load

    def load_records(self, conn: sqlite3.Connection, product_name: str, rows: Iterable[Mapping[str, Any]]) -> str:
        try:
            table_name = normalize_table_name(product_name)
            cleaned_rows = [clean_review_record(row) for row in rows]
            unique_rows = list(deduplicate_reviews(cleaned_rows))
            if not unique_rows:
                raise RetrievalError("No review records were available to store.")
            ensure_review_table(conn, table_name)
            inserted = insert_review_rows(conn, table_name, unique_rows)
            if inserted != len(unique_rows):
                raise RetrievalError(f"Inserted {inserted} of {len(unique_rows)} prepared review records.")
            if self.enrich_after_load:
                self.enrich_table(conn, table_name)
            return table_name
        except RetrievalError:
            raise
        except Exception as exc:  # noqa: BLE001 - retrieval should fail closed for the orchestrator.
            raise RetrievalError(f"Retrieval storage failed: {exc}") from exc

    def enrich_table(self, conn: sqlite3.Connection, table_name: str) -> None:
        try:
            self.language_agent.enrich_table(conn, table_name)
            self.topic_agent.enrich_table(conn, table_name)
            self.semantic_tagger.enrich_table(conn, table_name)
        except Exception as exc:  # noqa: BLE001 - preserve controlled failure boundary.
            raise RetrievalError(f"Review enrichment failed: {exc}") from exc

    def retrieve_live(self, product_name: str) -> str:
        if not self.settings.allow_live_retrieval:
            raise RuntimeError("Live retrieval is disabled. Set ALLOW_LIVE_RETRIEVAL=true only after approval.")
        raise NotImplementedError("Live retrieval adapter must be implemented with an approved source-specific protocol.")


def clean_review_record(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value.strip() if isinstance(value, str) else value
        for key, value in row.items()
    }


def deduplicate_reviews(rows: Iterable[Mapping[str, Any]]) -> Iterable[Mapping[str, Any]]:
    seen: set[str] = set()
    for row in rows:
        digest = hashlib.sha256(
            json.dumps(
                {
                    "asin": str(row.get("asin", "")).strip().lower(),
                    "title": str(row.get("title", "")).strip().lower(),
                    "content": str(row.get("content", "")).strip().lower(),
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        yield row

