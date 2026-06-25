from __future__ import annotations

import sqlite3

from llm_review_analysis.db.schema import validate_identifier
from llm_review_analysis.llm import LLMProvider


DEFAULT_TOPICS = ("battery", "delivery", "quality", "price", "usability")


class TopicAssignmentAgent:
    """LLM-based topic inference and per-review assignment.

    This intentionally uses "assignment" terminology because the current
    implementation infers labels and assigns reviews, rather than running an
    unsupervised clustering algorithm.
    """

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    def infer_topics(self, review_texts: list[str], *, max_topics: int = 10) -> list[str]:
        if not review_texts:
            return list(DEFAULT_TOPICS)
        prompt = "Infer 5-10 concise review topics as a comma-separated list:\n" + "\n".join(review_texts[:30])
        content = self.provider.generate(prompt, purpose="topic_list").content
        topics = [topic.strip().lower() for topic in content.split(",") if topic.strip()]
        return topics[:max_topics] or list(DEFAULT_TOPICS)

    def assign_topic(self, review_text: str, topics: list[str]) -> str:
        lower = review_text.lower()
        for topic in topics:
            if topic.lower() in lower:
                return topic
        prompt = f"Choose one topic from {', '.join(topics)} for this review:\n{review_text}"
        assigned = self.provider.generate(prompt, purpose="topic_assign").content.strip().lower()
        return assigned if assigned in topics else topics[0]

    def enrich_table(self, conn: sqlite3.Connection, table_name: str) -> int:
        table = validate_identifier(table_name)
        rows = conn.execute(f"SELECT id, title, content, translated_review FROM {table}").fetchall()
        texts = [
            " ".join(str(row[col] or "") for col in ("title", "translated_review", "content")).strip()
            for row in rows
        ]
        topics = self.infer_topics(texts)
        updates = [(self.assign_topic(text, topics), int(row["id"])) for row, text in zip(rows, texts)]
        if updates:
            conn.executemany(f"UPDATE {table} SET topic = ? WHERE id = ?", updates)
            conn.commit()
        return len(updates)
