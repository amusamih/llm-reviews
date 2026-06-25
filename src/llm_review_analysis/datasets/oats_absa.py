from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Iterable, Iterator, Mapping, Any
import xml.etree.ElementTree as ET

from llm_review_analysis.db.schema import validate_identifier


OATS_ANNOTATION_COLUMNS: tuple[str, ...] = (
    "dataset_id",
    "source_domain",
    "source_file",
    "review_id",
    "text",
    "opinion_index",
    "has_opinion",
    "target",
    "category",
    "entity",
    "attribute",
    "polarity",
    "start_offset",
    "end_offset",
)


@dataclass(frozen=True)
class ABSAOpinion:
    target: str
    category: str
    polarity: str
    start: str = ""
    end: str = ""


@dataclass(frozen=True)
class AnnotatedReview:
    review_id: str
    text: str
    opinions: tuple[ABSAOpinion, ...]
    source_file: str


def iter_oats_xml_reviews(path: str | Path) -> Iterator[AnnotatedReview]:
    """Yield SemEval/OATS-style sentence annotations from an XML file or directory."""

    path = Path(path)
    files = sorted(path.rglob("*.xml")) if path.is_dir() else [path]
    for file_path in files:
        root = ET.parse(file_path).getroot()
        source_domain = infer_oats_domain(str(file_path))
        for sentence_index, sentence in enumerate(root.findall(".//sentence"), start=1):
            text_node = sentence.find("text")
            text = (text_node.text or "").strip() if text_node is not None else ""
            if not text:
                continue
            opinion_nodes = list(sentence.findall(".//Opinion")) + list(sentence.findall(".//opinion"))
            opinions = tuple(
                ABSAOpinion(
                    target=opinion.attrib.get("target", ""),
                    category=opinion.attrib.get("category", ""),
                    polarity=opinion.attrib.get("polarity", ""),
                    start=opinion.attrib.get("from", ""),
                    end=opinion.attrib.get("to", ""),
                )
                for opinion in opinion_nodes
            )
            yield AnnotatedReview(
                review_id=sentence.attrib.get("id") or f"{source_domain}:{file_path.stem}:{sentence_index}",
                text=text,
                opinions=opinions,
                source_file=str(file_path),
            )


def oats_review_to_row(review: AnnotatedReview) -> dict[str, str]:
    """Map one annotated ABSA sentence into the project review-table schema."""

    categories = sorted({opinion.category for opinion in review.opinions if opinion.category})
    polarities = sorted({opinion.polarity for opinion in review.opinions if opinion.polarity})
    return {
        "asin": "oats_absa",
        "seller": "",
        "author": review.review_id,
        "rating": "",
        "title": "",
        "date": "",
        "country": "",
        "verified": "",
        "content": review.text,
        "language": "",
        "translated_review": "",
        "topic": ", ".join(categories),
        "semantic_tags": ", ".join(polarities),
    }


def oats_annotation_rows(
    review: AnnotatedReview,
    *,
    dataset_id: str = "oats_absa",
) -> list[dict[str, str | int]]:
    """Return opinion-level annotation rows for one OATS review/sentence.

    Reviews without opinions are preserved as a single row with
    ``has_opinion = 0`` so denominator counts remain explicit.
    """

    source_domain = infer_oats_domain(review.source_file)
    if not review.opinions:
        return [
            _annotation_row(
                review,
                dataset_id=dataset_id,
                source_domain=source_domain,
                opinion_index=-1,
                has_opinion=0,
            )
        ]
    return [
        _annotation_row(
            review,
            dataset_id=dataset_id,
            source_domain=source_domain,
            opinion=opinion,
            opinion_index=index,
            has_opinion=1,
        )
        for index, opinion in enumerate(review.opinions)
    ]


def ensure_oats_annotation_table(conn: sqlite3.Connection, table_name: str) -> None:
    table = validate_identifier(table_name)
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id TEXT,
            source_domain TEXT,
            source_file TEXT,
            review_id TEXT,
            text TEXT,
            opinion_index INTEGER,
            has_opinion INTEGER,
            target TEXT,
            category TEXT,
            entity TEXT,
            attribute TEXT,
            polarity TEXT,
            start_offset TEXT,
            end_offset TEXT
        )
        """
    )
    conn.commit()


def insert_oats_annotation_rows(
    conn: sqlite3.Connection,
    table_name: str,
    reviews: Iterable[AnnotatedReview],
    *,
    dataset_id: str = "oats_absa",
) -> int:
    table = validate_identifier(table_name)
    columns = OATS_ANNOTATION_COLUMNS
    placeholders = ", ".join("?" for _ in columns)
    column_sql = ", ".join(columns)
    rows: list[tuple[Any, ...]] = []
    for review in reviews:
        for row in oats_annotation_rows(review, dataset_id=dataset_id):
            rows.append(tuple(row[column] for column in columns))
    if not rows:
        return 0
    conn.executemany(f"INSERT INTO {table} ({column_sql}) VALUES ({placeholders})", rows)
    conn.commit()
    return len(rows)


def infer_oats_domain(source_file: str) -> str:
    normalized = source_file.replace("\\", "/").lower()
    if "/amazon_ff/" in normalized or "amazon_ff" in normalized:
        return "amazon_ff"
    if "/coursera/" in normalized or "coursera" in normalized:
        return "coursera"
    if "/hotels/" in normalized or "hotels" in normalized:
        return "hotels"
    return "unknown"


def split_oats_category(category: str) -> tuple[str, str]:
    if "#" not in category:
        return category, ""
    entity, attribute = category.split("#", 1)
    return entity, attribute


def _annotation_row(
    review: AnnotatedReview,
    *,
    dataset_id: str,
    source_domain: str,
    opinion_index: int,
    has_opinion: int,
    opinion: ABSAOpinion | None = None,
) -> dict[str, str | int]:
    category = opinion.category if opinion else ""
    entity, attribute = split_oats_category(category)
    return {
        "dataset_id": dataset_id,
        "source_domain": source_domain,
        "source_file": review.source_file,
        "review_id": review.review_id,
        "text": review.text,
        "opinion_index": opinion_index,
        "has_opinion": has_opinion,
        "target": opinion.target if opinion else "",
        "category": category,
        "entity": entity,
        "attribute": attribute,
        "polarity": opinion.polarity if opinion else "",
        "start_offset": opinion.start if opinion else "",
        "end_offset": opinion.end if opinion else "",
    }
