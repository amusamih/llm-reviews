from __future__ import annotations

import argparse
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_review_analysis.config import ensure_directories, load_settings
from llm_review_analysis.datasets import (
    APPROVED_DATASETS,
    amazon_review_to_row,
    ensure_oats_annotation_table,
    insert_oats_annotation_rows,
    iter_amazon_reviews,
    iter_oats_xml_reviews,
    oats_review_to_row,
)
from llm_review_analysis.db.schema import ensure_review_table, insert_review_rows, normalize_table_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare approved public review datasets for local experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List approved public datasets and their intended roles.")

    amazon = subparsers.add_parser("amazon", help="Load an Amazon Reviews 2023 JSONL/JSONL.GZ sample into SQLite.")
    amazon.add_argument("--input", required=True, type=Path, help="Path to a local Amazon Reviews 2023 .jsonl or .jsonl.gz file.")
    amazon.add_argument("--product-name", default="amazon_reviews_2023", help="SQLite table/product name to create.")
    amazon.add_argument("--limit", type=int, default=10000, help="Maximum number of JSONL rows to inspect.")
    amazon.add_argument("--database", type=Path, default=None, help="Optional SQLite output path.")

    oats = subparsers.add_parser("oats", help="Load OATS-ABSA XML annotations into SQLite.")
    oats.add_argument("--input", required=True, type=Path, help="Path to an OATS-ABSA XML file or directory.")
    oats.add_argument("--product-name", default="oats_absa", help="SQLite table/product name to create.")
    oats.add_argument("--limit", type=int, default=None, help="Maximum number of annotated rows to load.")
    oats.add_argument("--database", type=Path, default=None, help="Optional SQLite output path.")
    oats.add_argument("--annotations-table", default=None, help="Optional table for opinion-level OATS annotations.")

    args = parser.parse_args()
    if args.command == "list":
        _print_dataset_list()
    elif args.command == "amazon":
        _load_amazon(args.input, args.product_name, args.limit, args.database)
    elif args.command == "oats":
        _load_oats(args.input, args.product_name, args.limit, args.database, args.annotations_table)


def _print_dataset_list() -> None:
    for dataset in APPROVED_DATASETS.values():
        issues = ", ".join(dataset.evaluation_tags)
        print(f"{dataset.dataset_id}: {dataset.name}")
        print(f"  role: {dataset.role}")
        print(f"  source: {dataset.source_url}")
        print(f"  evaluation tags: {issues}")
        print(f"  notes: {dataset.notes}")


def _connect(database: Path | None) -> sqlite3.Connection:
    settings = load_settings()
    if database is not None:
        settings = settings.__class__(
            project_root=settings.project_root,
            database_path=database.resolve(),
            output_dir=settings.output_dir,
            vectorstore_dir=settings.vectorstore_dir,
            llm_provider=settings.llm_provider,
            llm_model=settings.llm_model,
            embedding_model=settings.embedding_model,
            allow_live_llm=settings.allow_live_llm,
            allow_live_retrieval=settings.allow_live_retrieval,
            log_level=settings.log_level,
        )
    ensure_directories(settings)
    conn = sqlite3.connect(settings.database_path)
    conn.row_factory = sqlite3.Row
    return conn


def _load_amazon(input_path: Path, product_name: str, limit: int, database: Path | None) -> None:
    rows = (amazon_review_to_row(record) for record in iter_amazon_reviews(input_path, limit=limit))
    _insert_rows(product_name, rows, database)


def _load_oats(
    input_path: Path,
    product_name: str,
    limit: int | None,
    database: Path | None,
    annotations_table: str | None,
) -> None:
    records = iter_oats_xml_reviews(input_path)
    if limit is not None:
        records = _take(records, limit)
    reviews = list(records)
    rows = (oats_review_to_row(record) for record in reviews)

    conn = _connect(database)
    try:
        table = normalize_table_name(product_name)
        ensure_review_table(conn, table)
        inserted = insert_review_rows(conn, table, rows)
        print(f"Inserted {inserted} rows into table '{table}' at {conn.execute('PRAGMA database_list').fetchone()[2]}")

        if annotations_table:
            annotation_table = normalize_table_name(annotations_table)
            ensure_oats_annotation_table(conn, annotation_table)
            annotation_rows = insert_oats_annotation_rows(conn, annotation_table, reviews)
            print(f"Inserted {annotation_rows} opinion annotation rows into table '{annotation_table}'")
    finally:
        conn.close()


def _insert_rows(product_name: str, rows, database: Path | None) -> None:
    conn = _connect(database)
    try:
        table = normalize_table_name(product_name)
        ensure_review_table(conn, table)
        inserted = insert_review_rows(conn, table, rows)
        print(f"Inserted {inserted} rows into table '{table}' at {conn.execute('PRAGMA database_list').fetchone()[2]}")
    finally:
        conn.close()


def _take(records, limit: int):
    count = 0
    for record in records:
        if count >= limit:
            break
        yield record
        count += 1


if __name__ == "__main__":
    main()

