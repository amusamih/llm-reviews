from __future__ import annotations

import argparse
from collections import Counter
import json
import sqlite3
import sys
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (SRC_ROOT, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from llm_review_analysis.agents.retrieval_agent import deduplicate_reviews
from llm_review_analysis.db.schema import REVIEW_COLUMNS, normalize_table_name, validate_identifier


DEFAULT_DATABASE = PROJECT_ROOT / "data" / "processed" / "evaluation_foundation.db"
DEFAULT_SOURCE_TABLE = "amazon_all_beauty_balanced_seeded"
DEFAULT_PRODUCT_NAME = "amazon all beauty"


def build_programmatic_gold_package(
    *,
    database_path: str | Path = DEFAULT_DATABASE,
    source_table: str = DEFAULT_SOURCE_TABLE,
    product_name: str = DEFAULT_PRODUCT_NAME,
    output_dir: str | Path = PROJECT_ROOT / "outputs" / "programmatic_gold",
    row_limit: int | None = None,
) -> dict[str, Path]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    source_table = validate_identifier(source_table)
    rows = _load_review_rows(database_path, source_table, row_limit=row_limit)
    rows = list(deduplicate_reviews(rows))
    if not rows:
        raise ValueError(f"No rows available in {source_table}")

    expected_table = normalize_table_name(product_name)
    prompt_items = build_programmatic_gold_prompts(
        rows,
        product_name=product_name,
        expected_table=expected_table,
        source_table=source_table,
    )
    reviews_path = output_root / "programmatic_gold_reviews.json"
    prompts_path = output_root / "programmatic_gold_prompts.json"
    manifest_path = output_root / "programmatic_gold_manifest.json"
    reviews_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    prompts_path.write_text(json.dumps(prompt_items, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "database_path": str(database_path),
        "source_table": source_table,
        "product_name": product_name,
        "expected_runtime_table": expected_table,
        "row_count": len(rows),
        "prompt_count": len(prompt_items),
        "gold_verification_status": "programmatically_verified",
        "gold_verification_method": "expected outputs computed directly from local SQLite rows before benchmark execution",
        "output_files": {
            "reviews": str(reviews_path),
            "prompts": str(prompts_path),
            "manifest": str(manifest_path),
        },
        "limitations": [
            "Gold is programmatically verified against the selected local table, not manually author-verified.",
            "Multilingual and translation-quality prompts are excluded because the selected Amazon pilot data has no language labels or reference translations.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"reviews": reviews_path, "prompts": prompts_path, "manifest": manifest_path}


def build_programmatic_gold_prompts(
    rows: list[dict[str, Any]],
    *,
    product_name: str,
    expected_table: str,
    source_table: str,
) -> list[dict[str, Any]]:
    count = len(rows)
    ratings = [_to_float(row.get("rating")) for row in rows if _to_float(row.get("rating")) is not None]
    avg_rating = round(mean(ratings), 2) if ratings else None
    rating_counts = dict(sorted(Counter(str(row.get("rating", "")) for row in rows).items()))
    dates = sorted(str(row.get("date", "")) for row in rows if str(row.get("date", "")).strip())
    start_date = dates[0] if dates else None
    end_date = dates[-1] if dates else None
    semantic_index, semantic_row, semantic_phrase = _select_semantic_source_row(rows)

    prompts: list[dict[str, Any]] = [
        _prompt(
            prompt_id="PGOLD-SQL-COUNT-001",
            category="direct_sql_factual",
            prompt_text=f"How many reviews for {product_name}?",
            product_name=product_name,
            expected_table=expected_table,
            expected_route="DIRECT_SQL",
            expected_result_type="text",
            expected_sql=f"SELECT COUNT(*) AS review_count FROM {expected_table}",
            expected_sql_pattern=rf"SELECT\s+COUNT\(\*\)\s+AS\s+review_count\s+FROM\s+{expected_table}",
            expected_answer_facts=[f"{count} reviews"],
            success_criteria=["route is DIRECT_SQL", "count comes from local SQLite rows"],
            gold_computation_method="sqlite_count",
            gold_source_query=f"SELECT COUNT(*) AS review_count FROM {source_table}",
            gold_source_records=[{"row_count": count}],
        ),
        _prompt(
            prompt_id="PGOLD-SQL-AVG-001",
            category="direct_sql_factual",
            prompt_text=f"What is the average rating for {product_name}?",
            product_name=product_name,
            expected_table=expected_table,
            expected_route="DIRECT_SQL",
            expected_result_type="text",
            expected_sql=f"SELECT AVG(CAST(rating AS REAL)) AS avg_rating FROM {expected_table}",
            expected_sql_pattern=r"AVG\((?:CAST\(rating\s+AS\s+REAL\)|rating)\)",
            expected_answer_facts=[f"{avg_rating:.2f}" if avg_rating is not None else "No numeric ratings"],
            success_criteria=["route is DIRECT_SQL", "average comes from local SQLite rows"],
            gold_computation_method="sqlite_avg_rating",
            gold_source_query=f"SELECT AVG(CAST(rating AS REAL)) AS avg_rating FROM {source_table}",
            gold_source_records=[{"avg_rating": avg_rating}],
        ),
        _prompt(
            prompt_id="PGOLD-SQL-VALIDATION-001",
            category="sql_generation_validation",
            prompt_text=f"Please count reviews for {product_name}, but do not delete anything.",
            product_name=product_name,
            expected_table=expected_table,
            expected_route="DIRECT_SQL",
            expected_result_type="text",
            expected_sql_pattern=r"^SELECT\s+COUNT\(\*\)",
            expected_answer_facts=[f"{count} reviews"],
            success_criteria=["generated SQL remains SELECT-only", "unsafe wording does not produce mutation SQL"],
            gold_computation_method="sqlite_count_with_unsafe_prompt_guard",
            gold_source_query=f"SELECT COUNT(*) AS review_count FROM {source_table}",
            gold_source_records=[{"row_count": count}],
        ),
        _prompt(
            prompt_id="PGOLD-SEMANTIC-EVIDENCE-001",
            category="semantic_reasoning",
            prompt_text=f"Why does review text mention {semantic_phrase} for {product_name}?",
            product_name=product_name,
            expected_table=expected_table,
            expected_route="SEMANTICS",
            expected_result_type="text",
            expected_answer_facts=[semantic_phrase],
            expected_source_review_ids=[str(semantic_index + 1)],
            expected_evidence_snippets=[semantic_phrase],
            success_criteria=["route is SEMANTICS", "evidence contains the computed source phrase"],
            gold_computation_method="source_record_text_match",
            gold_source_query=f"SELECT id, content FROM {source_table} WHERE id = {semantic_row.get('id', semantic_index + 1)}",
            gold_source_records=[
                {
                    "export_index": semantic_index,
                    "runtime_review_id": semantic_index + 1,
                    "source_content_snippet": str(semantic_row.get("content", ""))[:200],
                }
            ],
        ),
        _prompt(
            prompt_id="PGOLD-CHART-RATING-001",
            category="chart_numerical_consistency",
            prompt_text=f"Show the rating distribution for {product_name}",
            product_name=product_name,
            expected_table=expected_table,
            expected_route="ANALYTICS",
            expected_result_type="chart",
            expected_chart_type="bar",
            expected_chart_grouping="rating",
            expected_chart_values={key: float(value) for key, value in rating_counts.items()},
            expected_answer_facts=["bar chart"],
            success_criteria=["chart type is bar", "chart groups by rating", "chart values equal SQLite rating counts"],
            gold_computation_method="sqlite_group_by_rating",
            gold_source_query=f"SELECT rating, COUNT(*) AS value FROM {source_table} GROUP BY rating ORDER BY rating",
            gold_source_records=[{"rating": key, "count": value} for key, value in rating_counts.items()],
        ),
        _prompt(
            prompt_id="PGOLD-AMBIGUOUS-001",
            category="ambiguous_prompt",
            prompt_text=f"Tell me about {product_name}.",
            product_name=product_name,
            expected_table=expected_table,
            expected_route="SEMANTICS",
            expected_result_type="text",
            expected_failure_type="ambiguous_prompt",
            ambiguity_flag=True,
            success_criteria=["ambiguous prompt is recorded as a controlled failure or limitation"],
            gold_computation_method="programmatic_expected_failure",
            gold_source_query="not_applicable",
            gold_source_records=[{"expected_failure_type": "ambiguous_prompt"}],
        ),
        _prompt(
            prompt_id="PGOLD-MISSING-INFO-001",
            category="missing_information",
            prompt_text=f"Why is the warranty score low for {product_name}?",
            product_name=product_name,
            expected_table=expected_table,
            expected_route="SEMANTICS",
            expected_result_type="text",
            expected_evidence_snippets=["warranty"],
            expected_failure_type="missing_information",
            missing_information_flag=True,
            success_criteria=["system should not hallucinate missing warranty evidence"],
            gold_computation_method="programmatic_absence_check",
            gold_source_query=f"SELECT COUNT(*) AS warranty_mentions FROM {source_table} WHERE lower(content) LIKE '%warranty%'",
            gold_source_records=[{"warranty_mentions": _count_term(rows, "warranty")}],
        ),
        _prompt(
            prompt_id="PGOLD-FAILURE-UNKNOWN-PRODUCT-001",
            category="failure_case",
            prompt_text="How many reviews for unknown evaluation gadget?",
            product_name="unknown evaluation gadget",
            expected_table="unknown_evaluation_gadget",
            expected_route="DIRECT_SQL",
            expected_result_type="text",
            expected_sql_pattern="SELECT",
            expected_answer_facts=["0 reviews"],
            expected_failure_type="product_not_found",
            success_criteria=["unknown product is not silently treated as the selected product"],
            gold_computation_method="programmatic_expected_failure",
            gold_source_query="not_applicable",
            gold_source_records=[{"available_table": expected_table, "requested_table": "unknown_evaluation_gadget"}],
        ),
        _prompt(
            prompt_id="PGOLD-FAILURE-CHART-SCATTER-001",
            category="failure_case",
            prompt_text=f"Show a scatter plot of rating by date for {product_name}",
            product_name=product_name,
            expected_table=expected_table,
            expected_route="ANALYTICS",
            expected_result_type="chart",
            expected_chart_type="scatter",
            expected_chart_grouping="date",
            expected_failure_type="unsupported_chart_type",
            success_criteria=["unsupported chart request is recorded as a controlled failure"],
            gold_computation_method="programmatic_expected_failure",
            gold_source_query="not_applicable",
            gold_source_records=[{"supported_chart_types": ["bar", "line", "pie"]}],
        ),
        _prompt(
            prompt_id="PGOLD-TRANSLATION-UNSUPPORTED-001",
            category="translation_unsupported",
            prompt_text=f"Evaluate translation quality for {product_name} reviews.",
            product_name=product_name,
            expected_table=expected_table,
            expected_route="SEMANTICS",
            expected_result_type="text",
            expected_evidence_snippets=["reference translation"],
            expected_failure_type="translation_quality_not_evaluated",
            success_criteria=["translation-quality request is recorded as unsupported without reference translations"],
            gold_computation_method="programmatic_absence_check",
            gold_source_query="not_applicable_no_reference_translation_field_available",
            gold_source_records=[{"reference_translation_rows": 0}],
        ),
    ]
    if start_date and end_date and avg_rating is not None:
        date_avg = _average_rating_for_date_range(rows, start_date, end_date)
        prompts.insert(
            2,
            _prompt(
                prompt_id="PGOLD-SQL-DATE-RANGE-001",
                category="date_range_extraction",
                prompt_text=f"What is the average rating for {product_name} from {start_date} to {end_date}?",
                product_name=product_name,
                expected_table=expected_table,
                expected_route="DIRECT_SQL",
                expected_result_type="text",
                expected_date_range=f"{start_date}..{end_date}",
                expected_sql=(
                    f"SELECT AVG(CAST(rating AS REAL)) AS avg_rating FROM {expected_table} "
                    f"WHERE date >= '{start_date}' AND date <= '{end_date}'"
                ),
                expected_sql_pattern=(
                    r"WHERE\s+(?:date\s+BETWEEN\s+'[0-9]{4}-[0-9]{2}-[0-9]{2}'\s+AND\s+'[0-9]{4}-[0-9]{2}-[0-9]{2}'|"
                    r"date\s+>=\s+'[0-9]{4}-[0-9]{2}-[0-9]{2}'\s+AND\s+date\s+<=)"
                ),
                expected_answer_facts=[f"{date_avg:.2f}"],
                success_criteria=["date range is extracted", "average is computed over the selected date range"],
                gold_computation_method="sqlite_date_range_avg_rating",
                gold_source_query=(
                    f"SELECT AVG(CAST(rating AS REAL)) AS avg_rating FROM {source_table} "
                    f"WHERE date >= '{start_date}' AND date <= '{end_date}'"
                ),
                gold_source_records=[{"start_date": start_date, "end_date": end_date, "avg_rating": date_avg}],
            ),
        )
    return prompts


def _prompt(
    *,
    prompt_id: str,
    category: str,
    prompt_text: str,
    product_name: str,
    expected_table: str,
    expected_route: str,
    expected_result_type: str,
    expected_date_range: str | None = None,
    expected_sql: str | None = None,
    expected_sql_pattern: str | None = None,
    expected_answer_facts: list[str] | None = None,
    expected_source_review_ids: list[str] | None = None,
    expected_evidence_snippets: list[str] | None = None,
    expected_chart_type: str | None = None,
    expected_chart_values: dict[str, float] | None = None,
    expected_chart_grouping: str | None = None,
    expected_failure_type: str | None = None,
    ambiguity_flag: bool = False,
    missing_information_flag: bool = False,
    contradiction_flag: bool = False,
    success_criteria: list[str] | None = None,
    gold_computation_method: str = "",
    gold_source_query: str = "",
    gold_source_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "prompt_id": prompt_id,
        "category": category,
        "prompt_text": prompt_text,
        "language": "en",
        "product": product_name,
        "expected_product_table": expected_table,
        "expected_route": expected_route,
        "expected_date_range": expected_date_range,
        "expected_sql": expected_sql,
        "expected_sql_pattern": expected_sql_pattern,
        "expected_result_type": expected_result_type,
        "expected_answer_facts": expected_answer_facts or [],
        "expected_source_review_ids": expected_source_review_ids or [],
        "expected_evidence_snippets": expected_evidence_snippets or [],
        "expected_chart_type": expected_chart_type,
        "expected_chart_values": expected_chart_values or {},
        "expected_chart_grouping": expected_chart_grouping,
        "expected_failure_type": expected_failure_type,
        "ambiguity_flag": ambiguity_flag,
        "missing_information_flag": missing_information_flag,
        "contradiction_flag": contradiction_flag,
        "multiturn_context": False,
        "success_criteria": success_criteria or [],
        "gold_verification_status": "programmatically_verified",
        "gold_verified_by": None,
        "gold_computation_method": gold_computation_method,
        "gold_source_query": gold_source_query,
        "gold_source_records": gold_source_records or [],
        "gold_notes": "Expected value was computed programmatically from local repository data; no manual author verification claimed.",
        "evaluation_tags": ["benchmark", "baseline", "end-to-end"],
    }


def _load_review_rows(database_path: str | Path, source_table: str, *, row_limit: int | None) -> list[dict[str, Any]]:
    table = validate_identifier(source_table)
    limit_clause = f" LIMIT {int(row_limit)}" if row_limit is not None else ""
    with sqlite3.connect(database_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(f"SELECT * FROM {table} ORDER BY id{limit_clause}").fetchall()
    return [
        {column: str(row[column] or "") for column in REVIEW_COLUMNS if column != "id"}
        | {"source_id": str(row["id"])}
        for row in rows
    ]


def _select_semantic_source_row(rows: list[dict[str, Any]]) -> tuple[int, dict[str, Any], str]:
    preferred_terms = ("quality", "skin", "smell", "product", "good", "great")
    for term in preferred_terms:
        for index, row in enumerate(rows):
            content = str(row.get("content", ""))
            if term in content.lower():
                return index, row, _snippet_phrase(content, term)
    return 0, rows[0], _snippet_phrase(str(rows[0].get("content", "")), "")


def _snippet_phrase(content: str, preferred_term: str) -> str:
    words = content.replace(".", " ").replace(",", " ").split()
    if not words:
        return preferred_term or "review"
    lowered = [word.lower().strip("'\"") for word in words]
    if preferred_term and preferred_term in lowered:
        idx = lowered.index(preferred_term)
        start = max(0, idx - 2)
        end = min(len(words), idx + 3)
        return " ".join(words[start:end]).strip("'\"")
    return " ".join(words[:5]).strip("'\"")


def _average_rating_for_date_range(rows: list[dict[str, Any]], start_date: str, end_date: str) -> float:
    ratings = [
        rating
        for row in rows
        if start_date <= str(row.get("date", "")) <= end_date
        for rating in [_to_float(row.get("rating"))]
        if rating is not None
    ]
    return round(mean(ratings), 2) if ratings else 0.0


def _count_term(rows: list[dict[str, Any]], term: str) -> int:
    lowered = term.lower()
    return sum(1 for row in rows if lowered in str(row.get("content", "")).lower())


def _to_float(value: Any) -> float | None:
    try:
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a programmatically verified benchmark package from local SQLite rows.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--source-table", default=DEFAULT_SOURCE_TABLE)
    parser.add_argument("--product-name", default=DEFAULT_PRODUCT_NAME)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "programmatic_gold")
    parser.add_argument("--row-limit", type=int, default=None)
    args = parser.parse_args()
    paths = build_programmatic_gold_package(
        database_path=args.database,
        source_table=args.source_table,
        product_name=args.product_name,
        output_dir=args.output_dir,
        row_limit=args.row_limit,
    )
    for name, path in paths.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()

