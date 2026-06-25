from __future__ import annotations

import json
from pathlib import Path
import uuid

from evaluation.benchmark_metrics import latency_summary, normalize_sql
from evaluation.benchmark_schema import BenchmarkPrompt
from evaluation.run_benchmark import BENCHMARK_MODES, EVIDENCE_ID, load_benchmark_prompts, run_mock_benchmark


REQUIRED_PROMPT_FIELDS = {
    "prompt_id",
    "prompt_text",
    "category",
    "language",
    "expected_route",
    "expected_result_type",
    "expected_table",
    "evaluation_tags",
}


def test_expanded_benchmark_prompt_fixture_loads_required_categories_and_routes():
    raw_prompts = json.loads(Path("tests/fixtures/mock_benchmark_prompts.json").read_text(encoding="utf-8"))
    prompts = load_benchmark_prompts("tests/fixtures/mock_benchmark_prompts.json")

    assert len(prompts) == 17
    assert all(REQUIRED_PROMPT_FIELDS.issubset(raw_prompt) for raw_prompt in raw_prompts)
    assert all(isinstance(prompt, BenchmarkPrompt) for prompt in prompts)
    assert {prompt.expected_route for prompt in prompts} == {"DIRECT_SQL", "SEMANTICS", "ANALYTICS"}
    assert {prompt.category for prompt in prompts}.issuperset(
        {
            "direct_sql_factual",
            "semantic_reasoning",
            "data_analytics_chart",
            "multilingual_prompt",
            "product_extraction",
            "date_range_extraction",
            "sql_generation_validation",
            "answer_factual_consistency",
            "chart_type_correctness",
            "chart_numerical_consistency",
            "ambiguous_prompt",
            "missing_information",
            "contradictory_review",
            "failure_case",
            "multi_turn_context",
        }
    )


def test_benchmark_metrics_helpers():
    assert normalize_sql(" SELECT COUNT(*) AS review_count FROM sample_product; ") == (
        "select count(*) as review_count from sample_product"
    )
    summary = latency_summary([1.0, 2.0, 10.0])
    assert summary["count"] == 3
    assert summary["p50"] == 2.0
    assert summary["p95"] > summary["p50"]


def test_all_benchmark_modes_run_in_mock_and_write_summary_artifacts():
    artifacts = run_mock_benchmark(
        output_dir=_runtime_dir(),
        run_id="unit_mock_all_modes",
    )

    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    evidence = json.loads(artifacts.evidence_path.read_text(encoding="utf-8"))
    result_rows = _read_jsonl(artifacts.results_path)
    prompt_count = len(load_benchmark_prompts("tests/fixtures/mock_benchmark_prompts.json"))

    assert artifacts.manifest_path.exists()
    assert artifacts.results_path.exists()
    assert artifacts.summary_path.exists()
    assert artifacts.evidence_path.exists()
    assert manifest["live_mode"] is False
    assert manifest["mode"] == "mock"
    assert manifest["modes"] == list(BENCHMARK_MODES)
    assert manifest["prompt_count"] == prompt_count
    assert manifest["result_count"] == prompt_count * len(BENCHMARK_MODES)
    assert len(result_rows) == manifest["result_count"]
    assert summary["metrics"]["prompt_count"] == manifest["result_count"]
    assert summary["metrics"]["mode_count"] == len(BENCHMARK_MODES)
    assert set(summary["metrics"]["by_mode"]) == set(BENCHMARK_MODES)
    assert "routing_accuracy" in summary["metrics"]
    assert "sql_validity_rate" in summary["metrics"]
    assert "chart_numerical_consistency_rate" in summary["metrics"]
    assert "failure_rate_by_category" in summary["metrics"]
    assert summary["metrics"]["latency_ms"]["p50"] >= 0
    assert summary["metrics"]["latency_ms"]["p95"] >= summary["metrics"]["latency_ms"]["p50"]
    assert evidence["evidence_id"] == EVIDENCE_ID
    assert evidence["live_mock_status"] == "mock/offline; no live API calls"


def test_failure_cases_are_recorded_without_crashing_and_no_live_fields_are_populated():
    artifacts = run_mock_benchmark(
        output_dir=_runtime_dir(),
        run_id="unit_mock_failure_logging",
        benchmark_modes=("proposed_multi_agent", "sql_only", "single_agent_gpt4o_placeholder"),
    )
    rows = _read_jsonl(artifacts.results_path)

    assert rows
    assert any(row["failure_category"] for row in rows)
    assert any(row["failure_category"] == "placeholder_not_executed" for row in rows)
    assert any(row["prompt_id"] == "MOCK-MISSING-INFO-001" and row["failure_category"] for row in rows)
    assert all(row["model_provider"] == "MockLLMProvider" for row in rows)
    assert all(row["input_tokens"] is None for row in rows)
    assert all(row["output_tokens"] is None for row in rows)
    assert all(row["total_tokens"] is None for row in rows)
    assert all(row["estimated_cost_usd"] is None for row in rows)
    assert all(row["mode_execution_type"] == "mock" for row in rows)
    assert all(row["uses_live_gpt4o"] is False for row in rows)
    assert all(row["uses_mock_provider"] is True for row in rows)
    assert all(row["live_call_count"] == 0 for row in rows)


def test_proposed_mock_mode_records_expected_sql_evidence_and_chart_fields():
    artifacts = run_mock_benchmark(
        output_dir=_runtime_dir(),
        run_id="unit_mock_proposed_mode",
        benchmark_modes=("proposed_multi_agent",),
    )
    rows = _read_jsonl(artifacts.results_path)
    by_id = {row["prompt_id"]: row for row in rows}
    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))

    assert by_id["MOCK-SQL-COUNT-001"]["actual_sql"] == "SELECT COUNT(*) AS review_count FROM sample_product"
    assert by_id["MOCK-SQL-DATE-RANGE-001"]["actual_date_range"] == "2025-07-01..2025-07-03"
    assert by_id["MOCK-SEMANTIC-DELIVERY-001"]["evidence_ids"]
    assert by_id["MOCK-CHART-RATING-001"]["chart_path"]
    assert by_id["MOCK-CHART-RATING-001"]["chart_numerical_consistency"] is True
    assert summary["metrics"]["by_mode"]["proposed_multi_agent"]["prompt_count"] == 17
    assert summary["metrics"]["by_mode"]["proposed_multi_agent"]["chart_file_generation_rate"] is not None


def test_metrics_report_raw_adjusted_denominators_and_expected_refusal_success():
    artifacts = run_mock_benchmark(
        output_dir=_runtime_dir(),
        run_id="unit_mock_metric_denominators",
        benchmark_modes=("proposed_multi_agent",),
    )
    rows = _read_jsonl(artifacts.results_path)
    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    metrics = summary["metrics"]["by_mode"]["proposed_multi_agent"]

    assert "raw_aggregate_metrics" in metrics
    assert "eligibility_adjusted_metrics" in metrics
    assert "expected_refusal_metrics" in metrics
    assert "metric_denominators" in metrics
    assert metrics["raw_aggregate_metrics"]["sql_validity_rate"]["denominator"] >= (
        metrics["eligibility_adjusted_metrics"]["sql_validity_rate"]["denominator"]
    )
    assert metrics["raw_aggregate_metrics"]["chart_file_generation_rate"]["denominator"] >= (
        metrics["eligibility_adjusted_metrics"]["chart_file_generation_rate"]["denominator"]
    )
    assert metrics["expected_refusal_metrics"]["overall_expected_failure_handling_rate"]["denominator"] >= 1
    assert metrics["expected_refusal_metrics"]["overall_expected_failure_handling_rate"]["rate"] == 1.0
    assert metrics["overall_success_rate_with_expected_refusals"] == metrics["overall_success_rate"]

    expected_refusal_rows = [row for row in rows if row["expected_refusal_category"]]
    assert expected_refusal_rows
    assert all(row["expected_failure_handled"] is True for row in expected_refusal_rows)
    assert all(row["success"] is True for row in expected_refusal_rows)


def test_inapplicable_mode_rows_are_logged_separately_from_execution_failures():
    artifacts = run_mock_benchmark(
        output_dir=_runtime_dir(),
        run_id="unit_mock_inapplicable_modes",
        benchmark_modes=("sql_only", "sql_plus_semantic_retrieval"),
    )
    rows = _read_jsonl(artifacts.results_path)
    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    sql_only_semantic_rows = [
        row
        for row in rows
        if row["mode"] == "sql_only" and row["expected_route"] == "SEMANTICS"
    ]
    sql_plus_chart_rows = [
        row
        for row in rows
        if row["mode"] == "sql_plus_semantic_retrieval" and row["expected_route"] == "ANALYTICS"
    ]

    assert sql_only_semantic_rows
    assert sql_plus_chart_rows
    assert all(row["failure_category"] == "inapplicable_for_mode" for row in sql_only_semantic_rows)
    assert all(row["actual_result_type"] == "inapplicable" for row in sql_only_semantic_rows)
    assert all(row["inapplicable_reason"] for row in sql_only_semantic_rows)
    assert all(row["failure_category"] == "inapplicable_for_mode" for row in sql_plus_chart_rows)
    assert summary["metrics"]["by_mode"]["sql_only"]["inapplicable_count"] == len(sql_only_semantic_rows) + len(
        [row for row in rows if row["mode"] == "sql_only" and row["expected_route"] == "ANALYTICS"]
    )
    assert summary["metrics"]["by_mode"]["sql_plus_semantic_retrieval"]["inapplicable_count"] == len(sql_plus_chart_rows)
    assert summary["metrics"]["by_mode"]["sql_only"]["applicable_prompt_count"] < summary["metrics"]["by_mode"]["sql_only"]["prompt_count"]


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _runtime_dir() -> Path:
    path = Path("test_runtime") / f"benchmark_tests_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path

