from __future__ import annotations

import json
from pathlib import Path
import re
import sqlite3
from dataclasses import dataclass
from typing import Mapping

from llm_review_analysis.db.schema import validate_identifier
from llm_review_analysis.llm import LLMProvider

from .analysis_text import analysis_text_from_row

DEFAULT_TOPIC_DEFINITIONS: Mapping[str, str] = {
    "battery life": "Battery duration, charging endurance, or power drain during use.",
    "delivery": "Shipping speed, delivery condition, packaging, or arrival experience.",
    "quality": "Build quality, durability, defects, materials, or product finish.",
    "price": "Cost, value for money, discounts, or whether the price feels justified.",
    "usability": "Ease of setup, comfort, interface clarity, or day-to-day ease of use.",
    "camera quality": "Photo or video quality, camera features, or imaging performance.",
    "performance": "Speed, responsiveness, lag, processing power, or app performance.",
    "design": "Appearance, form factor, style, weight, or physical design.",
    "heating issue": "Overheating, excessive warmth, thermal throttling, or heat concerns.",
    "display/screen quality": "Screen brightness, clarity, resolution, color, or display defects.",
    "fast charging": "Charging speed, fast-charging behavior, or charging compatibility.",
    "user experience": "Overall experience, satisfaction, frustration, or general use impression.",
    "support": "Customer support, service responsiveness, replacement help, or assistance.",
    "compatibility": "Compatibility with devices, software, accessories, or standards.",
    "warranty": "Warranty claims, coverage, replacement, denial, or policy experience.",
}


class TopicAssignmentError(RuntimeError):
    def __init__(self, message: str, *, raw_responses: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.raw_responses = raw_responses


@dataclass(frozen=True)
class TopicAssignmentFailure:
    review_id: int | None
    reason: str
    raw_responses: tuple[str, ...]


class TopicAssignmentAgent:
    """LLM-based topic inference and per-review assignment.

    This intentionally uses "assignment" terminology because the current
    implementation infers labels and assigns reviews, rather than running an
    unsupervised clustering algorithm.
    """

    def __init__(
        self,
        provider: LLMProvider,
        *,
        max_assignment_retries: int = 2,
        topic_definitions: Mapping[str, str] | None = None,
    ) -> None:
        self.provider = provider
        self.max_assignment_retries = max(0, int(max_assignment_retries))
        self.topic_definitions = dict(topic_definitions or DEFAULT_TOPIC_DEFINITIONS)
        self.failures: list[TopicAssignmentFailure] = []

    def infer_topics(self, review_texts: list[str], *, max_topics: int = 10) -> list[str]:
        if not review_texts:
            raise TopicAssignmentError("No review texts were available for topic inference.")
        prompt = _topic_list_prompt(review_texts[:30], max_topics=max_topics)
        content = self.provider.generate(prompt, purpose="topic_list", response_format="json").content
        topics = _parse_topic_list(content)
        if not topics:
            raise TopicAssignmentError("Topic inference did not return a valid topic vocabulary.")
        return topics[:max_topics]

    def assign_topic(self, review_text: str, topics: list[str]) -> str:
        return self.assign_topic_with_trace(review_text, topics)["topic"]

    def assign_topic_with_trace(self, review_text: str, topics: list[str]) -> dict[str, object]:
        normalized_topics = _normalize_topic_list(topics)
        if not normalized_topics:
            raise TopicAssignmentError("No allowed topics were provided for topic assignment.")
        prompt = _topic_assignment_prompt(review_text, normalized_topics, self.topic_definitions)
        raw_responses: list[str] = []
        for attempt in range(self.max_assignment_retries + 1):
            response = self.provider.generate(prompt, purpose="topic_assign", response_format="json")
            raw = response.content.strip()
            raw_responses.append(raw)
            assigned = _parse_assigned_topic(raw, normalized_topics)
            if assigned is not None:
                return {
                    "topic": assigned,
                    "attempts": attempt + 1,
                    "raw_responses": tuple(raw_responses),
                    "model": response.model,
                    "usage": response.usage,
                }
        raise TopicAssignmentError(
            "Topic assignment failed after retries because the model did not return one allowed topic.",
            raw_responses=tuple(raw_responses),
        )

    def enrich_table(self, conn: sqlite3.Connection, table_name: str) -> int:
        table = validate_identifier(table_name)
        rows = conn.execute(f"SELECT id, title, content, translated_review, language FROM {table}").fetchall()
        texts = [analysis_text_from_row(row) for row in rows]
        topics = self.infer_topics(texts)
        updates: list[tuple[str | None, int]] = []
        failures: list[TopicAssignmentFailure] = []
        for row, text in zip(rows, texts):
            review_id = int(row["id"])
            try:
                updates.append((self.assign_topic(text, topics), review_id))
            except TopicAssignmentError as exc:
                failures.append(TopicAssignmentFailure(review_id, str(exc), exc.raw_responses))
                updates.append((None, review_id))
        if updates:
            conn.executemany(f"UPDATE {table} SET topic = ? WHERE id = ?", updates)
            conn.commit()
        self.failures.extend(failures)
        if failures:
            raise TopicAssignmentError(f"Topic assignment failed for {len(failures)} review(s).")
        return len(updates)


def _topic_list_prompt(review_texts: list[str], *, max_topics: int) -> str:
    joined = "\n".join(review_texts)
    return (
        f"Infer up to {max_topics} concise product-review topic labels from the reviews. "
        "Use short English noun phrases, avoid overlapping labels where possible, "
        "and return JSON only in the form {\"topics\": [\"label\", \"...\"]}.\n\n"
        f"Reviews:\n{joined}"
    )


def _topic_assignment_prompt(
    review_text: str,
    topics: list[str],
    topic_definitions: Mapping[str, str] | None = None,
) -> str:
    definitions = topic_definitions or DEFAULT_TOPIC_DEFINITIONS
    labels = "\n".join(f"- {topic}: {definitions.get(topic, f'The review is primarily about {topic}.')}" for topic in topics)
    return (
        "Assign exactly one dominant topic to the following product or service review. "
        "The review may be written in any language. Consider the review as a whole and select the allowed "
        "topic that best captures the review's primary praise, complaint, request, or evaluation. "
        "If several aspects are mentioned, choose the one that is most central to the user's evaluation. "
        "Do not select a topic merely because its words appear in the text. "
        "Do not invent new labels. Return JSON only in the form {\"topic\": \"one allowed topic\"}. "
        "Provide no explanation outside the structured response.\n\n"
        f"Allowed topic labels and definitions:\n{labels}\n\n"
        f"Review:\n{review_text}"
    )


def _parse_topic_list(content: str) -> list[str]:
    parsed: object | None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return []
    raw_topics: object
    if isinstance(parsed, dict):
        raw_topics = parsed.get("topics", parsed.get("labels", []))
    elif isinstance(parsed, list):
        raw_topics = parsed
    else:
        raw_topics = []
    if isinstance(raw_topics, list):
        candidates = raw_topics
    else:
        candidates = []
    return _normalize_topic_list([str(candidate) for candidate in candidates])


def _parse_assigned_topic(content: str, topics: list[str]) -> str | None:
    parsed: object | None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    raw: object
    if isinstance(parsed, dict):
        raw = parsed.get("topic")
    if not isinstance(raw, str):
        return None
    normalized = _normalize_topic(str(raw))
    topic_lookup = {_normalize_topic(topic): topic for topic in topics}
    if normalized in topic_lookup:
        return topic_lookup[normalized]
    return None


def _normalize_topic_list(topics: list[str] | tuple[str, ...]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for topic in topics:
        cleaned = _normalize_topic(str(topic))
        if cleaned and cleaned not in seen:
            normalized.append(cleaned)
            seen.add(cleaned)
    return normalized


def _normalize_topic(topic: str) -> str:
    cleaned = topic.strip().lower().replace("_", " ").replace("-", " ")
    cleaned = re.sub(r"[\[\]{}()\"']", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,;:")
    aliases = {
        "battery": "battery life",
        "battery duration": "battery life",
        "screen quality": "display/screen quality",
        "display quality": "display/screen quality",
        "overheating": "heating issue",
        "heating issues": "heating issue",
        "reliability": "charger reliability",
        "durability": "product durability",
        "connection issues": "connection stability",
        "product reliability": "charger reliability",
    }
    return aliases.get(cleaned, cleaned)


def load_topic_vocabulary(path: str | Path) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        raw_topics = payload
    elif isinstance(payload, dict):
        raw_topics = payload.get("topics", payload.get("labels", []))
    else:
        raw_topics = []
    topics = _normalize_topic_list([str(topic) for topic in raw_topics])
    if not topics:
        raise TopicAssignmentError(f"No topic labels were found in {path}.")
    return topics
