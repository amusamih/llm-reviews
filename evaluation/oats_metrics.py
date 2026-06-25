from __future__ import annotations

from collections import Counter
import sqlite3
from pathlib import Path
from typing import Literal

from llm_review_analysis.db.schema import validate_identifier


BaselineName = Literal["oracle", "majority", "empty"]


def load_oats_truth(
    database_path: str | Path,
    annotations_table: str,
    *,
    label_column: str,
) -> dict[str, set[str]]:
    table = validate_identifier(annotations_table)
    column = validate_identifier(label_column)
    truth: dict[str, set[str]] = {}
    with sqlite3.connect(database_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"SELECT review_id, has_opinion, {column} AS label FROM {table} ORDER BY review_id"
        ).fetchall()
    for row in rows:
        review_id = str(row["review_id"])
        truth.setdefault(review_id, set())
        label = str(row["label"] or "").strip()
        if int(row["has_opinion"] or 0) == 1 and label:
            truth[review_id].add(label)
    return truth


def build_baseline_predictions(
    truth: dict[str, set[str]],
    *,
    baseline: BaselineName,
    top_k: int = 1,
) -> dict[str, set[str]]:
    if baseline == "oracle":
        return {review_id: set(labels) for review_id, labels in truth.items()}
    if baseline == "empty":
        return {review_id: set() for review_id in truth}
    if baseline == "majority":
        most_common = [
            label
            for label, _ in Counter(label for labels in truth.values() for label in labels).most_common(top_k)
        ]
        return {review_id: set(most_common) for review_id in truth}
    raise ValueError(f"Unknown baseline: {baseline}")
