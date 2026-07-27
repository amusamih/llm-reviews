from __future__ import annotations

from collections.abc import Mapping
from typing import Any


ENGLISH_LANGUAGE_MARKERS = {"", "en", "eng", "english"}


def is_english_language(language: object) -> bool:
    return str(language or "").strip().lower() in ENGLISH_LANGUAGE_MARKERS


def build_analysis_text(
    *,
    title: object = "",
    content: object = "",
    translated_review: object = "",
    language: object = "",
) -> str:
    """Build the canonical production text used by enrichment agents."""

    title_text = str(title or "").strip()
    content_text = str(content or "").strip()
    translated_text = str(translated_review or "").strip()
    if is_english_language(language) or not translated_text:
        return " ".join(part for part in (title_text, content_text) if part).strip()
    return translated_text


def analysis_text_from_row(row: Mapping[str, Any]) -> str:
    return build_analysis_text(
        title=row["title"] if "title" in row.keys() else "",
        content=row["content"] if "content" in row.keys() else "",
        translated_review=row["translated_review"] if "translated_review" in row.keys() else "",
        language=row["language"] if "language" in row.keys() else "",
    )
