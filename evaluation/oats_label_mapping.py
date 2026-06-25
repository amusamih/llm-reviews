from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (SRC_ROOT, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from llm_review_analysis.db.schema import validate_identifier


SUPPORTED_LABEL_MAPPINGS: dict[str, dict[str, str]] = {
    "polarity": {
        "source_column": "polarity",
        "label_type": "sentiment polarity",
        "claim_scope": "OATS public sentiment/aspect evaluation only",
    },
    "category": {
        "source_column": "category",
        "label_type": "aspect category",
        "claim_scope": "OATS public aspect/category evaluation only",
    },
    "entity": {
        "source_column": "entity",
        "label_type": "aspect entity",
        "claim_scope": "OATS public aspect/entity evaluation only",
    },
    "attribute": {
        "source_column": "attribute",
        "label_type": "aspect attribute",
        "claim_scope": "OATS public aspect/attribute evaluation only",
    },
    "target": {
        "source_column": "target",
        "label_type": "opinion target",
        "claim_scope": "OATS public target extraction evaluation only",
    },
}

UNSUPPORTED_TARGETS: dict[str, str] = {
    "helpfulness": "OATS does not provide helpfulness labels.",
    "vagueness": "OATS does not provide vagueness/no-specific-justification labels.",
    "no_justification": "OATS does not provide no-justification labels.",
    "contradiction": "OATS does not provide contradiction labels.",
    "duplicate": "OATS does not provide duplicate labels; exact duplicates can only be computed programmatically.",
    "misleading_credibility": "OATS does not provide misleadingness or credibility labels.",
    "translation_quality": "OATS does not provide reference translations or bilingual quality judgments.",
    "routing": "Routing is a system benchmark concern, not an OATS annotation dimension.",
    "sql": "SQL correctness is a programmatic benchmark concern, not an OATS annotation dimension.",
    "chart_correctness": "Chart correctness is a programmatic benchmark concern, not an OATS annotation dimension.",
}


def oats_label_mapping_report() -> dict[str, Any]:
    return {
        "supported_label_mappings": SUPPORTED_LABEL_MAPPINGS,
        "unsupported_targets": UNSUPPORTED_TARGETS,
        "claim_boundary": (
            "Use OATS only for public aspect/opinion/sentiment-style label evaluation. "
            "Do not use OATS to claim routing, SQL, chart, translation-quality, IAA, "
            "helpfulness, contradiction, duplicate, or credibility performance."
        ),
    }


def validate_oats_label_dimension(label_dimension: str) -> dict[str, str]:
    normalized = label_dimension.strip().lower()
    if normalized in SUPPORTED_LABEL_MAPPINGS:
        return SUPPORTED_LABEL_MAPPINGS[normalized]
    if normalized in UNSUPPORTED_TARGETS:
        raise ValueError(f"Unsupported OATS label dimension '{label_dimension}': {UNSUPPORTED_TARGETS[normalized]}")
    raise ValueError(
        f"Unsupported OATS label dimension '{label_dimension}'. "
        f"Supported dimensions: {', '.join(sorted(SUPPORTED_LABEL_MAPPINGS))}."
    )


def load_oats_label_truth(
    database_path: str | Path,
    annotations_table: str,
    *,
    label_dimension: str,
) -> dict[str, set[str]]:
    mapping = validate_oats_label_dimension(label_dimension)
    table = validate_identifier(annotations_table)
    column = validate_identifier(mapping["source_column"])
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


def write_oats_label_mapping_report(output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(oats_label_mapping_report(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the supported/unsupported OATS label mapping report.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "oats" / "oats_label_mapping.json")
    args = parser.parse_args()
    path = write_oats_label_mapping_report(args.output)
    print(f"mapping_report={path}")


if __name__ == "__main__":
    main()
