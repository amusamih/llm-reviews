from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import date
import hashlib
import json
import sqlite3
from pathlib import Path
from statistics import mean, median
import sys
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_review_analysis.db.schema import REVIEW_COLUMNS, validate_identifier


SUMMARY_DISTRIBUTION_COLUMNS = ("language", "rating", "country", "verified", "topic", "semantic_tags")
SEMANTIC_DIMENSIONS = {
    "sentiment_polarity": {"positive", "negative"},
    "information_quality": {"helpful", "vague", "low-effort"},
    "credibility": {"potentially misleading", "misleading"},
    "similarity": {"duplicate"},
    "consistency": {"contradictory", "mixed"},
}


def compute_sqlite_dataset_stats(database_path: str | Path, table_name: str) -> dict[str, Any]:
    table = validate_identifier(table_name)
    with sqlite3.connect(database_path) as conn:
        conn.row_factory = sqlite3.Row
        total = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
        stats: dict[str, Any] = {"table": table, "total_reviews": total}
        for column in SUMMARY_DISTRIBUTION_COLUMNS:
            if column in REVIEW_COLUMNS:
                rows = conn.execute(
                    f"SELECT {column} AS value, COUNT(*) AS n FROM {table} GROUP BY {column} ORDER BY n DESC"
                ).fetchall()
                stats[f"{column}_distribution"] = {str(row["value"]): row["n"] for row in rows}
        stats["missing_counts"] = _missing_counts(conn, table)
        stats["nonempty_counts"] = {
            column: total - missing for column, missing in stats["missing_counts"].items()
        }
        stats["rating_summary"] = _numeric_summary(_column_values(conn, table, "rating"))
        stats["content_length_chars"] = _length_summary(_column_values(conn, table, "content"), unit="chars")
        stats["content_length_words"] = _length_summary(_column_values(conn, table, "content"), unit="words")
        stats["date_range"] = _date_range(_column_values(conn, table, "date"))
        rows = [dict(row) for row in conn.execute(f"SELECT {', '.join(REVIEW_COLUMNS)} FROM {table}").fetchall()]
        stats.update(
            _provenance_and_distribution_stats(
                rows,
                dataset_name=table,
                product_or_service=table,
                source_platform=None,
                source_format="sqlite",
                release_provenance_notes=[f"Computed from SQLite table {table}."],
            )
        )
    return stats


def compute_review_file_dataset_stats(
    input_path: str | Path,
    *,
    dataset_name: str | None = None,
    product_or_service: str | None = None,
    source_platform: str | None = None,
    release_provenance_notes: list[str] | None = None,
) -> dict[str, Any]:
    path = Path(input_path)
    rows = load_review_rows(path)
    return _provenance_and_distribution_stats(
        rows,
        dataset_name=dataset_name or path.stem,
        product_or_service=product_or_service,
        source_platform=source_platform,
        source_format=_source_format(path),
        release_provenance_notes=release_provenance_notes or [f"Computed from local fixture/source file {path}."],
    )


def load_review_rows(input_path: str | Path) -> list[dict[str, Any]]:
    path = Path(input_path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if suffix == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows
    if suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return [dict(row) for row in raw]
        if isinstance(raw, dict):
            for key in ("reviews", "rows", "data"):
                if isinstance(raw.get(key), list):
                    return [dict(row) for row in raw[key]]
        raise ValueError("JSON review input must be a list or contain a reviews/rows/data list")
    raise ValueError(f"Unsupported review input format: {path.suffix}")


def compute_annotation_stats(database_path: str | Path, table_name: str) -> dict[str, Any]:
    table = validate_identifier(table_name)
    with sqlite3.connect(database_path) as conn:
        conn.row_factory = sqlite3.Row
        total = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
        unique_reviews = conn.execute(f"SELECT COUNT(DISTINCT review_id) AS n FROM {table}").fetchone()["n"]
        opinion_rows = conn.execute(f"SELECT COUNT(*) AS n FROM {table} WHERE has_opinion = 1").fetchone()["n"]
        no_opinion_rows = conn.execute(f"SELECT COUNT(*) AS n FROM {table} WHERE has_opinion = 0").fetchone()["n"]
        stats: dict[str, Any] = {
            "table": table,
            "total_annotation_rows": total,
            "unique_reviews": unique_reviews,
            "opinion_rows": opinion_rows,
            "no_opinion_rows": no_opinion_rows,
        }
        for column in ("source_domain", "polarity", "category", "entity", "attribute"):
            stats[f"{column}_distribution"] = _distribution(conn, table, column)
        rows = conn.execute(
            f"SELECT review_id, SUM(CASE WHEN has_opinion = 1 THEN 1 ELSE 0 END) AS n "
            f"FROM {table} GROUP BY review_id"
        ).fetchall()
        stats["opinions_per_review"] = _number_summary([row["n"] for row in rows])
    return stats


def _provenance_and_distribution_stats(
    rows: Iterable[Mapping[str, Any]],
    *,
    dataset_name: str,
    product_or_service: str | None,
    source_platform: str | None,
    source_format: str,
    release_provenance_notes: list[str],
) -> dict[str, Any]:
    row_list = [dict(row) for row in rows]
    total = len(row_list)
    requires_author_input: dict[str, str] = {}
    real_generated_counts, real_generated_input_needed = _real_generated_counts(row_list)
    if real_generated_input_needed:
        requires_author_input["real_vs_custom_synthetic_generated_counts"] = (
            "No reliable row-level origin/provenance field was found. Authors must provide real/custom/synthetic/generated counts and generation procedure."
        )
    if not source_platform:
        requires_author_input["source_platform"] = "Source platform is not encoded in the available fixture rows."
    if not product_or_service:
        requires_author_input["product_or_service"] = "Product/service name must be confirmed by the authors."

    missing_values = _missing_counts_from_rows(row_list)
    semantic_labels = _semantic_label_distribution(row_list)
    return {
        "dataset_name": dataset_name,
        "source_format": source_format,
        "total_reviews": total,
        "product_or_service": product_or_service or "requires_author_input",
        "source_platform": source_platform or "requires_author_input",
        "real_vs_custom_synthetic_generated_counts": real_generated_counts,
        "language_distribution": _distribution_from_rows(row_list, "language"),
        "rating_distribution": _distribution_from_rows(row_list, "rating"),
        "country_distribution": _distribution_from_rows(row_list, "country"),
        "date_distribution": _distribution_from_rows(row_list, "date"),
        "topic_distribution": _distribution_from_rows(row_list, "topic"),
        "semantic_label_distribution": semantic_labels,
        "multi_label_semantic_dimension_distribution": _semantic_dimension_distribution(semantic_labels),
        "duplicate_count": _duplicate_summary(row_list)["duplicate_count"],
        "duplicate_summary": _duplicate_summary(row_list),
        "missing_values": missing_values,
        "missing_counts": missing_values,
        "review_length_distribution": {
            "chars": _length_summary([str(_row_value(row, "content")) for row in row_list], unit="chars"),
            "words": _length_summary([str(_row_value(row, "content")) for row in row_list], unit="words"),
        },
        "release_provenance_notes": release_provenance_notes,
        "requires_author_input": requires_author_input,
    }


def _distribution(conn: sqlite3.Connection, table: str, column: str) -> dict[str, int]:
    rows = conn.execute(
        f"SELECT {column} AS value, COUNT(*) AS n FROM {table} GROUP BY {column} ORDER BY n DESC"
    ).fetchall()
    return {str(row["value"]): row["n"] for row in rows}


def _distribution_from_rows(rows: list[Mapping[str, Any]], column: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        value = str(_row_value(row, column)).strip()
        if value:
            counts[value] += 1
        else:
            counts["<missing>"] += 1
    return dict(sorted(counts.items()))


def _semantic_label_distribution(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        raw = str(_row_value(row, "semantic_tags") or _row_value(row, "semantic_labels")).strip()
        if not raw:
            counts["<missing>"] += 1
            continue
        for label in _split_labels(raw):
            counts[label] += 1
    return dict(sorted(counts.items()))


def _semantic_dimension_distribution(label_counts: Mapping[str, int]) -> dict[str, int]:
    dimensions: Counter[str] = Counter()
    for label, count in label_counts.items():
        if label == "<missing>":
            dimensions["<missing>"] += count
            continue
        matched = False
        normalized = label.lower()
        for dimension, labels in SEMANTIC_DIMENSIONS.items():
            if normalized in labels:
                dimensions[dimension] += count
                matched = True
        if not matched:
            dimensions["other"] += count
    return dict(sorted(dimensions.items()))


def _duplicate_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    digests: Counter[str] = Counter()
    for row in rows:
        key = {
            "asin": str(_row_value(row, "asin")).strip().lower(),
            "title": str(_row_value(row, "title")).strip().lower(),
            "content": str(_row_value(row, "content")).strip().lower(),
        }
        digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()
        digests[digest] += 1
    duplicate_groups = sum(1 for count in digests.values() if count > 1)
    duplicate_count = sum(count - 1 for count in digests.values() if count > 1)
    return {
        "duplicate_count": duplicate_count,
        "duplicate_groups": duplicate_groups,
        "unique_review_fingerprints": len(digests),
        "method": "exact SHA-256 over normalized asin/title/content",
    }


def _missing_counts_from_rows(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    columns = [column for column in REVIEW_COLUMNS if column != "id"]
    counts = {column: 0 for column in columns}
    for row in rows:
        for column in columns:
            value = _row_value(row, column)
            if value is None or str(value).strip() == "":
                counts[column] += 1
    return counts


def _real_generated_counts(rows: list[Mapping[str, Any]]) -> tuple[dict[str, int], bool]:
    candidate_columns = ("review_origin", "origin", "source_type", "provenance", "synthetic", "generated", "is_generated")
    present_column = next((column for column in candidate_columns if any(column in row for row in rows)), None)
    if not present_column:
        return {"requires_author_input": len(rows)}, True
    counts: Counter[str] = Counter()
    for row in rows:
        raw = str(row.get(present_column, "")).strip().lower()
        if raw in {"1", "true", "yes", "generated", "synthetic", "custom", "llm"}:
            counts["custom_synthetic_generated"] += 1
        elif raw in {"0", "false", "no", "real", "organic", "scraped", "public"}:
            counts["real_naturally_occurring"] += 1
        elif raw:
            counts[raw] += 1
        else:
            counts["requires_author_input"] += 1
    return dict(sorted(counts.items())), False


def _split_labels(raw: str) -> list[str]:
    normalized = raw.replace(";", ",").replace("|", ",")
    return [label.strip().lower() for label in normalized.split(",") if label.strip()]


def _row_value(row: Mapping[str, Any], column: str) -> Any:
    if column in row:
        return row[column]
    aliases = {
        "content": ("review", "review_text", "text"),
        "rating": ("stars", "overall"),
        "date": ("review_date", "timestamp"),
        "country": ("marketplace",),
        "semantic_tags": ("semantic_labels", "labels"),
    }
    for alias in aliases.get(column, ()):
        if alias in row:
            return row[alias]
    return ""


def _source_format(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    return suffix or "unknown"


def _missing_counts(conn: sqlite3.Connection, table: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for column in REVIEW_COLUMNS:
        if column == "id":
            continue
        row = conn.execute(
            f"SELECT SUM(CASE WHEN {column} IS NULL OR TRIM(CAST({column} AS TEXT)) = '' THEN 1 ELSE 0 END) AS n "
            f"FROM {table}"
        ).fetchone()
        counts[column] = int(row["n"] or 0)
    return counts


def _column_values(conn: sqlite3.Connection, table: str, column: str) -> list[str]:
    rows = conn.execute(f"SELECT {column} AS value FROM {table}").fetchall()
    return ["" if row["value"] is None else str(row["value"]) for row in rows]


def _numeric_summary(values: list[str]) -> dict[str, float | int | None]:
    numbers: list[float] = []
    invalid = 0
    for value in values:
        if not value.strip():
            continue
        try:
            numbers.append(float(value))
        except ValueError:
            invalid += 1
    summary = _number_summary(numbers)
    summary["invalid_count"] = invalid
    return summary


def _number_summary(values: list[float | int]) -> dict[str, float | int | None]:
    numbers = [float(value) for value in values]
    if not numbers:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p95": None,
        }
    sorted_numbers = sorted(numbers)
    p95_index = min(len(sorted_numbers) - 1, int(round((len(sorted_numbers) - 1) * 0.95)))
    return {
        "count": len(sorted_numbers),
        "min": sorted_numbers[0],
        "max": sorted_numbers[-1],
        "mean": round(mean(sorted_numbers), 4),
        "median": round(median(sorted_numbers), 4),
        "p95": round(sorted_numbers[p95_index], 4),
    }


def _length_summary(values: list[str], *, unit: str) -> dict[str, float | int | None]:
    if unit == "words":
        lengths = [len(value.split()) for value in values if value.strip()]
    else:
        lengths = [len(value) for value in values if value.strip()]
    return _number_summary(lengths)


def _date_range(values: list[str]) -> dict[str, str | int | None]:
    parsed: list[date] = []
    invalid = 0
    for value in values:
        value = value.strip()
        if not value:
            continue
        try:
            parsed.append(date.fromisoformat(value[:10]))
        except ValueError:
            invalid += 1
    if not parsed:
        return {
            "valid_count": 0,
            "invalid_count": invalid,
            "min": None,
            "max": None,
        }
    return {
        "valid_count": len(parsed),
        "invalid_count": invalid,
        "min": min(parsed).isoformat(),
        "max": max(parsed).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Write review-table dataset statistics as JSON.")
    parser.add_argument("--database", type=Path, help="SQLite database path.")
    parser.add_argument("--table", help="Review table name.")
    parser.add_argument("--input", type=Path, help="CSV, JSON, or JSONL review fixture/source file.")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON path.")
    parser.add_argument("--annotations-table", default=None, help="Optional opinion-level annotation table.")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--product-or-service", default=None)
    parser.add_argument("--source-platform", default=None)
    args = parser.parse_args()

    if args.input:
        stats = compute_review_file_dataset_stats(
            args.input,
            dataset_name=args.dataset_name,
            product_or_service=args.product_or_service,
            source_platform=args.source_platform,
        )
    elif args.database and args.table:
        stats = compute_sqlite_dataset_stats(args.database, args.table)
    else:
        raise SystemExit("Provide either --input or both --database and --table.")
    if args.annotations_table and args.database:
        stats["annotation_stats"] = compute_annotation_stats(args.database, args.annotations_table)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    label = args.table or args.input or stats.get("dataset_name", "dataset")
    print(f"Wrote stats for {label} ({stats['total_reviews']} reviews) to {args.output}")


if __name__ == "__main__":
    main()
