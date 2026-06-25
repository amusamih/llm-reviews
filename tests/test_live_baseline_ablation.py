from __future__ import annotations

from pathlib import Path
import uuid

from evaluation.live_baseline_ablation import (
    DEFAULT_MODE_NAMES,
    build_mode_comparison,
    estimate_live_operations,
    resolve_live_modes,
)
from evaluation.run_benchmark import load_gold_benchmark_prompts, run_mock_benchmark
from evaluation.benchmark_metrics import summarize_benchmark_results
from evaluation.benchmark_schema import BenchmarkCheck, BenchmarkResult


def test_live_baseline_ablation_resolves_only_approved_bounded_modes():
    specs = resolve_live_modes(["all"])
    names = [spec.mode.name for spec in specs]

    assert names == list(DEFAULT_MODE_NAMES)
    assert "single_agent_gpt4o_placeholder" not in names
    assert "single_agent_rag_placeholder" not in names
    assert specs[0].mode.name == "proposed_multi_agent_live_gpt4o"
    assert specs[0].uses_live_gpt4o is True
    assert specs[0].uses_faiss is True
    assert next(spec for spec in specs if spec.mode.name == "no_vector_retrieval").uses_faiss is False
    assert next(spec for spec in specs if spec.mode.name == "sql_only").mode.supported_routes == ("DIRECT_SQL",)


def test_live_operation_estimate_marks_inapplicable_rows_and_embedding_operations():
    prompts = load_gold_benchmark_prompts(
        "outputs/programmatic_gold/amazon_all_beauty_20260624/programmatic_gold_prompts.json"
    )
    specs = resolve_live_modes(["proposed_multi_agent_live_gpt4o", "sql_only", "no_vector_retrieval"])
    estimate = estimate_live_operations(prompts, specs)

    assert estimate["total_estimated_tracked_gpt4o_calls"] > 0
    assert estimate["total_estimated_embedding_batch_operations"] > 0
    assert estimate["by_mode"]["sql_only"]["inapplicable_rows"] > 0
    assert estimate["by_mode"]["no_vector_retrieval"].get("openai_embedding_batch_operations", 0) == 0
    assert estimate["by_mode"]["no_vector_retrieval"]["deterministic_lexical_semantic_rows"] > 0


def test_mode_comparison_contains_required_reporting_columns():
    artifacts = run_mock_benchmark(
        output_dir=_runtime_dir(),
        run_id="unit_live_mode_comparison_source",
        benchmark_modes=("proposed_multi_agent",),
    )
    rows = [_result_from_row(row) for row in _read_jsonl(artifacts.results_path)]
    metrics = summarize_benchmark_results(rows)
    comparison = build_mode_comparison(metrics, rows, resolve_live_modes(["proposed_multi_agent_live_gpt4o"]))

    assert comparison
    row = comparison[0]
    assert "overall_success_rate_with_expected_refusals" in row
    assert "sql_validity_rate_sql_eligible" in row
    assert "expected_failure_handling_rate" in row
    assert "live_call_count" in row
    assert "mock_call_count" in row
    assert "deterministic_call_count" in row


def _read_jsonl(path):
    import json

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _result_from_row(row):
    payload = dict(row)
    payload["checks"] = tuple(BenchmarkCheck(**check) for check in payload.get("checks", ()))
    return BenchmarkResult(**payload)


def _runtime_dir() -> Path:
    path = Path("test_runtime") / f"live_baseline_tests_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path
