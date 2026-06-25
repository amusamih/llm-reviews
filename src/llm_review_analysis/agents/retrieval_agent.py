from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Iterable, Mapping, Any

from llm_review_analysis.config import Settings
from llm_review_analysis.db.schema import ensure_review_table, insert_review_rows, normalize_table_name


class RetrievalAgent:
    """Offline-first retrieval/persistence agent.

    Live scraping is intentionally disabled here. Live retrieval experiments
    should use fixture data or an explicitly approved retrieval adapter.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def load_records(self, conn: sqlite3.Connection, product_name: str, rows: Iterable[Mapping[str, Any]]) -> str:
        table_name = normalize_table_name(product_name)
        unique_rows = list(deduplicate_reviews(rows))
        ensure_review_table(conn, table_name)
        insert_review_rows(conn, table_name, unique_rows)
        return table_name

    def retrieve_live(self, product_name: str) -> str:
        if not self.settings.allow_live_retrieval:
            raise RuntimeError("Live retrieval is disabled. Set ALLOW_LIVE_RETRIEVAL=true only after approval.")
        raise NotImplementedError("Live retrieval adapter must be implemented with an approved source-specific protocol.")


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

