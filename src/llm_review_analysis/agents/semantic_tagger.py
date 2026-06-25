from __future__ import annotations

from dataclasses import dataclass
import sqlite3

from llm_review_analysis.db.schema import validate_identifier


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
    def __init__(self, taxonomy: SemanticTaxonomy | None = None) -> None:
        self.taxonomy = taxonomy or SemanticTaxonomy()

    def tag_text(self, text: str) -> list[str]:
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
