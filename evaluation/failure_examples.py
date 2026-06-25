from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MITIGATIONS = {
    "ambiguous_prompt": "Clarify that ambiguous prompts should trigger clarification or be reported as a limitation.",
    "context_missing": "Report missing context rather than inferring unstated conversational history.",
    "missing_information": "Return insufficient-evidence behavior rather than hallucinating unsupported facts.",
    "product_not_found": "Do not silently map unknown products to another product table.",
    "translation_quality_not_evaluated": "State that translation quality requires reference translations or bilingual assessment.",
    "unsupported_chart_type": "Constrain chart requests to supported deterministic chart specifications.",
    "unsupported_route_for_mode": "Treat mode limitations as ablation/baseline behavior, not system failure.",
}


def collect_failure_examples(results_path: str | Path, *, max_examples: int = 12) -> dict[str, Any]:
    rows = _read_jsonl(results_path)
    failures = [
        row
        for row in rows
        if row.get("failure_category") or row.get("expected_failure_type") or row.get("success") is False
    ]
    examples = []
    seen_categories: set[str] = set()
    for row in failures:
        category = str(row.get("expected_failure_type") or row.get("failure_category") or "unclassified_failure")
        if category in seen_categories and len(examples) >= max_examples:
            continue
        seen_categories.add(category)
        examples.append(_example_from_row(row, category))
        if len(examples) >= max_examples and len(seen_categories) >= 6:
            break
    return {
        "source_results_path": str(results_path),
        "total_result_rows": len(rows),
        "failure_result_rows": len(failures),
        "example_count": len(examples),
        "examples": examples,
        "coverage": sorted(seen_categories),
        "notes": [
            "Examples are derived from benchmark result rows, not manually fabricated.",
            "Mock/programmatic benchmark examples are controlled system-behavior evidence only.",
        ],
    }


def write_failure_examples(results_path: str | Path, output_path: str | Path, *, max_examples: int = 12) -> Path:
    payload = collect_failure_examples(results_path, max_examples=max_examples)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _example_from_row(row: dict[str, Any], category: str) -> dict[str, Any]:
    checks = row.get("checks") or []
    failed_checks = [check.get("name") for check in checks if not check.get("passed")]
    return {
        "prompt_id": row.get("prompt_id"),
        "prompt": row.get("prompt_text"),
        "mode": row.get("mode"),
        "expected_behavior": {
            "expected_failure_type": row.get("expected_failure_type"),
            "success_criteria": row.get("success_criteria"),
            "expected_route": row.get("expected_route"),
            "expected_table": row.get("expected_table"),
        },
        "actual_system_behavior": {
            "actual_route": row.get("actual_route"),
            "actual_table": row.get("actual_table"),
            "actual_result_type": row.get("actual_result_type"),
            "response_preview": row.get("response_preview"),
            "failed_checks": failed_checks,
        },
        "failure_category": category,
        "failure_reason": row.get("failure_reason") or row.get("failure_message"),
        "mitigation_or_reporting_limitation": MITIGATIONS.get(
            category,
            "Record as a controlled failure and avoid unsupported performance claims.",
        ),
    }


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract controlled failure examples from benchmark results.")
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-examples", type=int, default=12)
    args = parser.parse_args()
    path = write_failure_examples(args.results, args.output, max_examples=args.max_examples)
    print(f"failure_examples={path}")


if __name__ == "__main__":
    main()
