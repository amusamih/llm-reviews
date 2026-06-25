from __future__ import annotations

import re
import sqlite3
from typing import Iterable, Sequence

from .schema import REVIEW_COLUMNS, validate_identifier


class SQLValidationError(ValueError):
    pass


FORBIDDEN_KEYWORDS = {
    "alter",
    "attach",
    "create",
    "delete",
    "detach",
    "drop",
    "insert",
    "pragma",
    "replace",
    "update",
    "vacuum",
}

SQL_KEYWORDS = {
    "and",
    "as",
    "asc",
    "avg",
    "between",
    "by",
    "case",
    "cast",
    "coalesce",
    "count",
    "date",
    "desc",
    "distinct",
    "else",
    "end",
    "from",
    "group",
    "having",
    "in",
    "is",
    "like",
    "limit",
    "lower",
    "max",
    "min",
    "not",
    "null",
    "or",
    "order",
    "round",
    "select",
    "strftime",
    "sum",
    "then",
    "when",
    "where",
}

IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
FROM_JOIN_RE = re.compile(r"\b(?:from|join)\s+([A-Za-z_][A-Za-z0-9_]*)\b", re.IGNORECASE)


def _strip_comments(sql: str) -> str:
    sql = re.sub(r"--.*?$", "", sql, flags=re.MULTILINE)
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    return sql.strip()


def _strip_string_literals(sql: str) -> str:
    return re.sub(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"", " ", sql)


def _ensure_single_statement(sql: str) -> None:
    stripped = sql.strip()
    if not stripped:
        raise SQLValidationError("SQL is empty")
    semicolon_positions = [idx for idx, char in enumerate(stripped) if char == ";"]
    if not semicolon_positions:
        return
    if semicolon_positions != [len(stripped) - 1]:
        raise SQLValidationError("Only one SELECT statement is allowed")


def validate_select_sql(
    sql: str,
    *,
    allowed_tables: Iterable[str],
    allowed_columns: Iterable[str] = REVIEW_COLUMNS,
) -> str:
    cleaned = _strip_comments(sql)
    _ensure_single_statement(cleaned)
    cleaned = cleaned.rstrip(";").strip()
    lowered = cleaned.lower()
    if not lowered.startswith("select "):
        raise SQLValidationError("Only SELECT statements are allowed")

    no_strings = _strip_string_literals(lowered)
    tokens = set(IDENTIFIER_RE.findall(no_strings))
    forbidden = tokens.intersection(FORBIDDEN_KEYWORDS)
    if forbidden:
        raise SQLValidationError(f"Forbidden SQL keyword(s): {', '.join(sorted(forbidden))}")

    allowed_table_set = {validate_identifier(table) for table in allowed_tables}
    allowed_column_set = set(allowed_columns)
    table_refs = {match.group(1) for match in FROM_JOIN_RE.finditer(cleaned)}
    if not table_refs:
        raise SQLValidationError("SELECT must reference an allowed table")
    unknown_tables = table_refs.difference(allowed_table_set)
    if unknown_tables:
        raise SQLValidationError(f"Unknown table(s): {', '.join(sorted(unknown_tables))}")

    allowed_identifiers = allowed_table_set | allowed_column_set | SQL_KEYWORDS
    aliases = _extract_aliases(cleaned)
    unknown_identifiers = {
        token
        for token in tokens
        if token not in allowed_identifiers and token not in aliases and not token.isdigit()
    }
    if unknown_identifiers:
        raise SQLValidationError(f"Unknown identifier(s): {', '.join(sorted(unknown_identifiers))}")
    return cleaned


def _extract_aliases(sql: str) -> set[str]:
    aliases = set(re.findall(r"\bas\s+([A-Za-z_][A-Za-z0-9_]*)\b", sql, flags=re.IGNORECASE))
    aliases.update(re.findall(r"\b(?:from|join)\s+[A-Za-z_][A-Za-z0-9_]*\s+([A-Za-z_][A-Za-z0-9_]*)\b", sql, flags=re.IGNORECASE))
    return {alias.lower() for alias in aliases}


def execute_validated_select(
    conn: sqlite3.Connection,
    sql: str,
    *,
    allowed_tables: Sequence[str],
    allowed_columns: Sequence[str] = REVIEW_COLUMNS,
) -> tuple[list[str], list[tuple]]:
    cleaned = validate_select_sql(sql, allowed_tables=allowed_tables, allowed_columns=allowed_columns)
    cursor = conn.execute(cleaned)
    columns = [description[0] for description in cursor.description or []]
    rows = [tuple(row) for row in cursor.fetchall()]
    return columns, rows
