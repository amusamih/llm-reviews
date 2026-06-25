from __future__ import annotations

from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


def iter_amazon_reviews(path: str | Path, *, limit: int | None = None) -> Iterator[dict[str, Any]]:
    """Yield raw Amazon Reviews 2023 JSON objects from .jsonl or .jsonl.gz files."""

    if limit is not None and limit <= 0:
        return
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    count = 0
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if limit is not None and count >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
            count += 1


def amazon_review_to_row(record: Mapping[str, Any]) -> dict[str, str]:
    """Map one Amazon Reviews 2023 record into the project review-table schema."""

    return {
        "asin": _clean(record.get("parent_asin") or record.get("asin")),
        "seller": _clean(record.get("store")),
        "author": _clean(record.get("user_id")),
        "rating": _clean(record.get("rating")),
        "title": _clean(record.get("title")),
        "date": _timestamp_to_date(record.get("timestamp")),
        "country": "",
        "verified": _clean(record.get("verified_purchase")),
        "content": _clean(record.get("text")),
        "language": "",
        "translated_review": "",
        "topic": "",
        "semantic_tags": "",
    }


def amazon_records_to_rows(records: Iterable[Mapping[str, Any]]) -> Iterator[dict[str, str]]:
    for record in records:
        row = amazon_review_to_row(record)
        if row["content"]:
            yield row


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _timestamp_to_date(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        timestamp = float(value)
    except (TypeError, ValueError):
        return _clean(value)
    if timestamp > 10_000_000_000:
        timestamp = timestamp / 1000.0
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).date().isoformat()
