from __future__ import annotations

import re
import sqlite3
from typing import Iterable, Mapping, Any


REVIEW_COLUMNS: tuple[str, ...] = (
    "id",
    "asin",
    "seller",
    "author",
    "rating",
    "title",
    "date",
    "country",
    "verified",
    "content",
    "language",
    "translated_review",
    "topic",
    "semantic_tags",
)

WRITABLE_REVIEW_COLUMNS = tuple(col for col in REVIEW_COLUMNS if col != "id")
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_identifier(identifier: str) -> str:
    if not IDENTIFIER_RE.fullmatch(identifier):
        raise ValueError(f"Unsafe SQL identifier: {identifier!r}")
    return identifier


def normalize_table_name(name: str) -> str:
    normalized = re.sub(r"\W+", "_", name.strip().lower()).strip("_")
    if not normalized:
        raise ValueError("Table name cannot be empty after normalization")
    return validate_identifier(normalized)


def ensure_review_table(conn: sqlite3.Connection, table_name: str) -> None:
    table = validate_identifier(table_name)
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asin TEXT,
            seller TEXT,
            author TEXT,
            rating TEXT,
            title TEXT,
            date TEXT,
            country TEXT,
            verified TEXT,
            content TEXT,
            language TEXT,
            translated_review TEXT,
            topic TEXT,
            semantic_tags TEXT
        )
        """
    )
    conn.commit()


def insert_review_rows(conn: sqlite3.Connection, table_name: str, rows: Iterable[Mapping[str, Any]]) -> int:
    table = validate_identifier(table_name)
    columns = WRITABLE_REVIEW_COLUMNS
    placeholders = ", ".join("?" for _ in columns)
    column_sql = ", ".join(columns)
    values = [tuple(str(row.get(col, "")) for col in columns) for row in rows]
    if not values:
        return 0
    conn.executemany(f"INSERT INTO {table} ({column_sql}) VALUES ({placeholders})", values)
    conn.commit()
    return len(values)


def list_review_tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
    return [row[0] for row in rows if row[0] != "sqlite_sequence"]
