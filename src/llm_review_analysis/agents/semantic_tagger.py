from __future__ import annotations

import json
import re
from dataclasses import dataclass
import sqlite3
from typing import Any

from llm_review_analysis.db.schema import validate_identifier
from llm_review_analysis.llm import LLMProvider

from .analysis_text import analysis_text_from_row


class SemanticTaggingError(RuntimeError):
    def __init__(self, message: str, *, raw_responses: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.raw_responses = raw_responses


@dataclass(frozen=True)
class SemanticTaggingFailure:
    review_id: int | None
    reason: str
    raw_responses: tuple[str, ...]


@dataclass(frozen=True)
class SemanticTaxonomy:
    sentiment_polarity: tuple[str, ...] = ("positive", "negative")
    information_quality: tuple[str, ...] = ("helpful", "vague", "no justification")
    consistency: tuple[str, ...] = ("contradictory",)
    duplication: tuple[str, ...] = ("duplicate",)
    credibility: tuple[str, ...] = ("potentially misleading",)

    @property
    def all_labels(self) -> tuple[str, ...]:
        return (
            self.sentiment_polarity
            + self.information_quality
            + self.consistency
            + self.duplication
            + self.credibility
        )


class SemanticTagger:
    def __init__(
        self,
        taxonomy: SemanticTaxonomy | None = None,
        *,
        provider: LLMProvider | None = None,
        use_provider: bool = False,
        max_retries: int = 2,
    ) -> None:
        self.taxonomy = taxonomy or SemanticTaxonomy()
        self.provider = provider
        self.use_provider = use_provider
        self.max_retries = max(0, int(max_retries))
        self.failures: list[SemanticTaggingFailure] = []

    def tag_text(self, text: str) -> list[str]:
        if self.use_provider:
            if self.provider is None:
                raise SemanticTaggingError("Provider-backed semantic tagging was requested without a provider.")
            return self._tag_text_with_provider(text)
        return self._deterministic_tags(text)

    def tag_text_with_trace(self, text: str) -> dict[str, Any]:
        prompt = _semantic_tagging_prompt(text, self.taxonomy.all_labels)
        raw_responses: list[str] = []
        for attempt in range(self.max_retries + 1):
            response = self.provider.generate(prompt, purpose="semantic_tagging", response_format="json")
            raw = response.content.strip()
            raw_responses.append(raw)
            tags = _parse_semantic_tags(raw, self.taxonomy.all_labels)
            if tags is not None:
                return {
                    "semantic_tags": tags,
                    "attempts": attempt + 1,
                    "raw_responses": tuple(raw_responses),
                    "model": response.model,
                    "usage": response.usage,
                }
        raise SemanticTaggingError(
            "Semantic tagging failed after retries because the model did not return valid allowed labels.",
            raw_responses=tuple(raw_responses),
        )

    def _tag_text_with_provider(self, text: str) -> list[str]:
        return list(self.tag_text_with_trace(text)["semantic_tags"])

    def _deterministic_tags(self, text: str) -> list[str]:
        lower = text.lower()
        tags: list[str] = []
        if any(word in lower for word in ("great", "excellent", "love", "perfect", "good")):
            tags.append("positive")
        if any(word in lower for word in ("bad", "poor", "broken", "waste", "terrible", "failed")):
            tags.append("negative")
        if len(lower.split()) >= 8:
            tags.append("helpful")
        if len(lower.split()) <= 4:
            tags.append("vague")
        if any(phrase in lower for phrase in ("no reason", "not sure", "just bad", "just good")):
            tags.append("no justification")
        if any(phrase in lower for phrase in ("but", "however", "although")) and {"positive", "negative"}.issubset(tags):
            tags.append("contradictory")
        if "copy" in lower or "same review" in lower:
            tags.append("duplicate")
        if "not as advertised" in lower or "misleading" in lower:
            tags.append("potentially misleading")
        return [tag for tag in tags if tag in self.taxonomy.all_labels]

    def enrich_table(self, conn: sqlite3.Connection, table_name: str) -> int:
        table = validate_identifier(table_name)
        rows = conn.execute(f"SELECT id, title, content, translated_review, language FROM {table}").fetchall()
        updates: list[tuple[str | None, int]] = []
        failures: list[SemanticTaggingFailure] = []
        for row in rows:
            review_id = int(row["id"])
            text = analysis_text_from_row(row)
            try:
                tags = ", ".join(self.tag_text(text))
                updates.append((tags, review_id))
            except SemanticTaggingError as exc:
                failures.append(SemanticTaggingFailure(review_id, str(exc), exc.raw_responses))
                updates.append((None, review_id))
        if updates:
            conn.executemany(f"UPDATE {table} SET semantic_tags = ? WHERE id = ?", updates)
            conn.commit()
        self.failures.extend(failures)
        if failures:
            raise SemanticTaggingError(f"Semantic tagging failed for {len(failures)} review(s).")
        return len(updates)


def _semantic_tagging_prompt(text: str, allowed_labels: tuple[str, ...]) -> str:
    definitions = {
        "positive": "clear satisfaction, praise, or favorable evaluation",
        "negative": "clear dissatisfaction, complaint, or unfavorable evaluation",
        "helpful": "specific information that can help a buyer understand use, quality, or trade-offs",
        "vague": "generic, unclear, or too little detail to support a buyer decision",
        "no justification": "sentiment or judgment is given without supporting explanation",
        "contradictory": "internally mixed or conflicting positive and negative claims",
        "duplicate": "appears copied, repeated, templated, or substantially reused",
        "potentially misleading": "may misrepresent the product, discuss the wrong product, or rely on unrealistic expectations",
    }
    labels = "\n".join(f"- {label}: {definitions.get(label, label)}" for label in allowed_labels)
    return (
        "Assign review-level semantic tags to the following product or service review. "
        "The review may be written in any language. Infer the review meaning directly and apply the same "
        "criteria regardless of the input language. Select every allowed label that applies and no labels that do not apply. "
        "Use only the allowed labels. Return JSON only in the form "
        '{"semantic_tags": ["label", "..."]}.\n\n'
        f"Allowed labels and definitions:\n{labels}\n\n"
        f"Review:\n{text}"
    )
def _parse_semantic_tags(content: str, allowed_labels: tuple[str, ...]) -> list[str] | None:
    raw_tags: object
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        raw_tags = parsed.get("semantic_tags", parsed.get("tags"))
    elif isinstance(parsed, list):
        raw_tags = parsed
    elif isinstance(parsed, str):
        raw_tags = parsed
    elif parsed is None and "," in content:
        raw_tags = content.split(",")
    elif parsed is None:
        raw_tags = [content]
    else:
        return None

    if isinstance(raw_tags, str):
        candidates = raw_tags.split(",")
    elif isinstance(raw_tags, list):
        candidates = raw_tags
    else:
        return None

    tags = _normalize_tags(candidates, allowed_labels)
    if not tags and parsed is None:
        return None
    return tags


def _normalize_tags(candidates: list[object], allowed_labels: tuple[str, ...]) -> list[str]:
    allowed = {label.lower(): label for label in allowed_labels}
    aliases = {
        "positive sentiment": "positive",
        "negative sentiment": "negative",
        "informative": "helpful",
        "specific": "helpful",
        "unclear": "vague",
        "generic": "vague",
        "nojustification": "no justification",
        "no justification": "no justification",
        "not justified": "no justification",
        "without justification": "no justification",
        "contradiction": "contradictory",
        "conflicting": "contradictory",
        "copied": "duplicate",
        "repeated": "duplicate",
        "misleading": "potentially misleading",
        "potentially misleading": "potentially misleading",
    }
    tags: list[str] = []
    for candidate in candidates:
        normalized = re.sub(r"\s+", " ", str(candidate).strip().lower().replace("_", " ").replace("-", " "))
        normalized = aliases.get(normalized, normalized)
        label = allowed.get(normalized)
        if label is not None and label not in tags:
            tags.append(label)
    return tags
