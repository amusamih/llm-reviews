from __future__ import annotations

import json
import re
from dataclasses import dataclass
import sqlite3

from llm_review_analysis.db.schema import validate_identifier
from llm_review_analysis.llm import LLMProvider


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
    ) -> None:
        self.taxonomy = taxonomy or SemanticTaxonomy()
        self.provider = provider
        self.use_provider = use_provider

    def tag_text(self, text: str) -> list[str]:
        if self.use_provider and self.provider is not None:
            provider_tags = self._tag_text_with_provider(text)
            if provider_tags is not None:
                return provider_tags
        return self._deterministic_tags(text)

    def _tag_text_with_provider(self, text: str) -> list[str] | None:
        prompt = _semantic_tagging_prompt(text, self.taxonomy.all_labels)
        try:
            response = self.provider.generate(prompt, purpose="semantic_tagging", response_format="json")
        except Exception:  # noqa: BLE001 - semantic enrichment falls back to the offline-safe tagger.
            return None
        return _parse_semantic_tags(response.content, self.taxonomy.all_labels)

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
        rows = conn.execute(f"SELECT id, title, content, translated_review FROM {table}").fetchall()
        updates = []
        for row in rows:
            text = " ".join(str(row[col] or "") for col in ("title", "content", "translated_review"))
            tags = ", ".join(self.tag_text(text))
            updates.append((tags, int(row["id"])))
        if updates:
            conn.executemany(f"UPDATE {table} SET semantic_tags = ? WHERE id = ?", updates)
            conn.commit()
        return len(updates)


def _semantic_tagging_prompt(text: str, allowed_labels: tuple[str, ...]) -> str:
    labels = "\n".join(f"- {label}" for label in allowed_labels)
    return (
        "Assign review-level semantic tags to the following product or service review. "
        "Use only the allowed labels. Return JSON only in the form "
        '{"semantic_tags": ["label", "..."]}.\n\n'
        f"Allowed labels:\n{labels}\n\n"
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
        "nojustification": "no justification",
        "no justification": "no justification",
        "not justified": "no justification",
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
