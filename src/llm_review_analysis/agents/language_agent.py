from __future__ import annotations

import re
import sqlite3

from llm_review_analysis.db.schema import REVIEW_COLUMNS, validate_identifier
from llm_review_analysis.llm import LLMProvider


class LanguageAgent:
    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    def detect_and_translate_text(self, text: str) -> tuple[str, str]:
        response = self.provider.generate(
            _language_prompt(text),
            purpose="language",
        ).content.strip()
        language = _extract_line(response, "LANGUAGE") or "en"
        translation = _extract_line(response, "TRANSLATION") or text
        return language.strip(), translation.strip()

    def translate_text(self, text: str, target_language: str) -> str:
        trace = self.translate_text_with_trace(text, target_language)
        return str(trace["translation"])

    def translate_text_with_trace(
        self,
        text: str,
        target_language: str,
        *,
        fallback_to_source: bool = True,
    ) -> dict[str, object]:
        response = self.provider.generate(
            _translate_prompt(text, target_language),
            purpose="translation",
        )
        raw = response.content.strip()
        parsed = _extract_line(raw, "TRANSLATION")
        translation = (parsed or raw).strip()
        parser_status = "translation_line" if parsed is not None else "raw_response_without_translation_prefix"
        if not translation:
            parser_status = "empty_response"
            if fallback_to_source:
                translation = text
        return {
            "translation": translation,
            "raw_response": raw,
            "parser_status": parser_status,
            "model": response.model,
            "usage": response.usage,
        }

    def enrich_table(self, conn: sqlite3.Connection, table_name: str) -> int:
        table = validate_identifier(table_name)
        rows = conn.execute(f"SELECT id, title, content FROM {table}").fetchall()
        updates: list[tuple[str, str, int]] = []
        for row in rows:
            review_text = " ".join(str(row[col] or "") for col in ("title", "content")).strip()
            if not review_text:
                continue
            language, translation = self.detect_and_translate_text(review_text)
            updates.append((language, translation, int(row["id"])))
        if updates:
            conn.executemany(f"UPDATE {table} SET language = ?, translated_review = ? WHERE id = ?", updates)
            conn.commit()
        return len(updates)


def _language_prompt(text: str) -> str:
    return (
        "Detect the language of the review and translate it to English only if needed.\n"
        "Return lines in this exact format:\n"
        "LANGUAGE: <iso-code>\n"
        "TRANSLATION: <English text, omit if already English>\n"
        f"Review:\n{text}"
    )


def _translate_prompt(text: str, target_language: str) -> str:
    return (
        f"Translate only the following text to {target_language}.\n"
        "Preserve praise, criticism, uncertainty, negation, intensity, quantities, and product references. "
        "Do not summarize, explain, classify, add topic labels, add semantic tags, or add interpretation.\n"
        "Return one line in this exact format:\n"
        "TRANSLATION: <translated text>\n"
        f"Text:\n{text}"
    )


def _extract_line(text: str, key: str) -> str | None:
    match = re.search(rf"^{re.escape(key)}\s*:\s*(.+)$", text, flags=re.IGNORECASE | re.MULTILINE)
    return match.group(1).strip() if match else None
