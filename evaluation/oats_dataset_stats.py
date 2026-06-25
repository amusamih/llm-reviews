from __future__ import annotations

import argparse
from collections import Counter
import json
import sqlite3
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (SRC_ROOT, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from evaluation.oats_label_mapping import UNSUPPORTED_TARGETS, oats_label_mapping_report
from llm_review_analysis.db.schema import validate_identifier


DEFAULT_DATABASE = PROJECT_ROOT / "data" / "processed" / "evaluation_foundation.db"
DEFAULT_REVIEWS_TABLE = "oats_amazon_finefood"
DEFAULT_ANNOTATIONS_TABLE = "oats_amazon_finefood_opinions"


def compute_oats_dataset_stats(
    database_path: str | Path = DEFAULT_DATABASE,
    *,
    reviews_table: str = DEFAULT_REVIEWS_TABLE,
    annotations_table: str = DEFAULT_ANNOTATIONS_TABLE,
) -> dict[str, Any]:
    reviews_table = validate_identifier(reviews_table)
    annotations_table = validate_identifier(annotations_table)
    database = Path(database_path)
    with sqlite3.connect(database) as conn:
        conn.row_factory = sqlite3.Row
        review_rows = conn.execute(f"SELECT * FROM {reviews_table} ORDER BY id").fetchall()
        annotation_rows = conn.execute(f"SELECT * FROM {annotations_table} ORDER BY id").fetchall()

    contents = [str(row["content"] or "") for row in review_rows]
    normalized_contents = [_normalize_text(text) for text in contents if text.strip()]
    duplicate_count = len(normalized_contents) - len(set(normalized_contents))
    product_ids = sorted({str(row["asin"] or "") for row in review_rows if str(row["asin"] or "").strip()})
    lengths_words = [len(text.split()) for text in contents]
    lengths_chars = [len(text) for text in contents]

    opinion_rows = [row for row in annotation_rows if int(row["has_opinion"] or 0) == 1]
    no_opinion_rows = [row for row in annotation_rows if int(row["has_opinion"] or 0) == 0]
    source_files = sorted({str(row["source_file"] or "") for row in annotation_rows if str(row["source_file"] or "").strip()})

    return {
        "dataset_name": "OATS-ABSA Amazon FineFood",
        "database_path": str(database),
        "reviews_table": reviews_table,
        "annotations_table": annotations_table,
        "review_count": len(review_rows),
        "opinion_annotation_row_count": len(annotation_rows),
        "opinion_rows": len(opinion_rows),
        "no_opinion_rows": len(no_opinion_rows),
        "product_or_item_count_available": bool(product_ids),
        "product_or_item_count": len(product_ids),
        "product_or_item_ids": product_ids[:20],
        "source_platform": "OATS-ABSA public dataset; Amazon FineFood domain",
        "source_files": source_files,
        "rating_distribution": _counter_from_rows(review_rows, "rating"),
        "language_distribution": _counter_from_rows(review_rows, "language"),
        "review_length_words": _numeric_summary(lengths_words),
        "review_length_chars": _numeric_summary(lengths_chars),
        "aspect_category_distribution": _counter_from_rows(annotation_rows, "category", opinion_only=True),
        "aspect_entity_distribution": _counter_from_rows(annotation_rows, "entity", opinion_only=True),
        "aspect_attribute_distribution": _counter_from_rows(annotation_rows, "attribute", opinion_only=True),
        "sentiment_polarity_distribution": _counter_from_rows(annotation_rows, "polarity", opinion_only=True),
        "missing_values_reviews": _missing_counts(review_rows),
        "missing_values_annotations": _missing_counts(annotation_rows),
        "duplicate_count_by_exact_normalized_content": duplicate_count,
        "label_mapping": oats_label_mapping_report(),
        "unsupported_evaluation_targets": UNSUPPORTED_TARGETS,
        "limitations": [
            "OATS Amazon FineFood has public final labels, but no local annotator IDs or multiple annotator label sets.",
            "IAA cannot be computed from the local files.",
            "Ratings, dates, countries, languages, and source URLs are not available in the prepared OATS review rows.",
            "Use OATS for aspect/category/polarity-style evaluation only, not routing, SQL, chart, translation-quality, or user-study claims.",
        ],
    }


def write_oats_dataset_stats(
    output_path: str | Path,
    *,
    database_path: str | Path = DEFAULT_DATABASE,
    reviews_table: str = DEFAULT_REVIEWS_TABLE,
    annotations_table: str = DEFAULT_ANNOTATIONS_TABLE,
) -> Path:
    stats = compute_oats_dataset_stats(
        database_path,
        reviews_table=reviews_table,
        annotations_table=annotations_table,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _counter_from_rows(rows: Iterable[sqlite3.Row], column: str, *, opinion_only: bool = False) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        if opinion_only and "has_opinion" in row.keys() and int(row["has_opinion"] or 0) != 1:
            continue
        value = str(row[column] or "").strip()
        counts[value] += 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _missing_counts(rows: list[sqlite3.Row]) -> dict[str, int]:
    if not rows:
        return {}
    columns = list(rows[0].keys())
    return {
        column: sum(1 for row in rows if str(row[column] or "").strip() == "")
        for column in columns
    }


def _numeric_summary(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0, "median": 0.0, "p95": 0.0}
    ordered = sorted(values)
    return {
        "count": len(values),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": round(mean(ordered), 3),
        "median": round(median(ordered), 3),
        "p95": round(_percentile(ordered, 95), 3),
    }


def _percentile(sorted_values: list[int], percentile: float) -> float:
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = (percentile / 100.0) * (len(sorted_values) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = rank - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute OATS Amazon FineFood dataset statistics.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--reviews-table", default=DEFAULT_REVIEWS_TABLE)
    parser.add_argument("--annotations-table", default=DEFAULT_ANNOTATIONS_TABLE)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "oats" / "oats_dataset_stats_20260624.json")
    args = parser.parse_args()
    path = write_oats_dataset_stats(
        args.output,
        database_path=args.database,
        reviews_table=args.reviews_table,
        annotations_table=args.annotations_table,
    )
    print(f"stats={path}")


if __name__ == "__main__":
    main()
