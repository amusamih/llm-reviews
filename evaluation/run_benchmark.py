from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
import re
import sqlite3
import sys
import time
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (SRC_ROOT, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from evaluation.benchmark_metrics import contains_all, normalize_sql, summarize_benchmark_results
from evaluation.benchmark_schema import (
    BenchmarkArtifacts,
    BenchmarkCheck,
    BenchmarkPrompt,
    BenchmarkResult,
    BenchmarkRunManifest,
)
from llm_review_analysis.agents import RetrievalAgent, ReviewOrchestrator
from llm_review_analysis.config import ensure_directories, load_settings
from llm_review_analysis.db.schema import REVIEW_COLUMNS, normalize_table_name
from llm_review_analysis.db.sql_validator import SQLValidationError, validate_select_sql
from llm_review_analysis.llm import MockLLMProvider


DEFAULT_PROMPTS_PATH = PROJECT_ROOT / "tests" / "fixtures" / "mock_benchmark_prompts.json"
DEFAULT_REVIEWS_PATH = PROJECT_ROOT / "tests" / "fixtures" / "mock_benchmark_reviews.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "benchmarks"
BENCHMARK_EVALUATION_TAGS = ("routing", "safety", "charting", "benchmark", "baseline", "end-to-end")
EVIDENCE_ID = "EVID-BENCHMARK-MODES-001"


@dataclass(frozen=True)
class BenchmarkMode:
    name: str
    description: str
    route_strategy: str = "orchestrator"
    supported_routes: tuple[str, ...] = ("DIRECT_SQL", "SEMANTICS", "ANALYTICS")
    disabled_fields: tuple[str, ...] = ()
    placeholder: bool = False


BENCHMARK_MODES: dict[str, BenchmarkMode] = {
    "proposed_multi_agent": BenchmarkMode(
        name="proposed_multi_agent",
        description="Paper-aligned multi-agent mock path with orchestrator routing.",
    ),
    "rule_based_routing_baseline": BenchmarkMode(
        name="rule_based_routing_baseline",
        description="Deterministic keyword-routing baseline using the same local tools.",
        route_strategy="rule_based",
    ),
    "sql_only": BenchmarkMode(
        name="sql_only",
        description="Baseline that forces every prompt through the SQL path.",
        route_strategy="sql_only",
        supported_routes=("DIRECT_SQL",),
    ),
    "sql_plus_semantic_retrieval": BenchmarkMode(
        name="sql_plus_semantic_retrieval",
        description="Baseline with SQL and semantic retrieval but no analytics/chart path.",
        supported_routes=("DIRECT_SQL", "SEMANTICS"),
    ),
    "no_semantic_tags": BenchmarkMode(
        name="no_semantic_tags",
        description="Ablation that clears semantic tag fields before local retrieval.",
        disabled_fields=("semantic_tags",),
    ),
    "no_topic_tags": BenchmarkMode(
        name="no_topic_tags",
        description="Ablation that clears topic fields before local retrieval.",
        disabled_fields=("topic",),
    ),
    "no_translation": BenchmarkMode(
        name="no_translation",
        description="Ablation that clears translated text and language metadata.",
        disabled_fields=("translated_review", "language"),
    ),
    "no_vector_retrieval": BenchmarkMode(
        name="no_vector_retrieval",
        description="Ablation interface for disabling vector retrieval; mock mode remains lexical/offline.",
    ),
    "no_orchestrator_simplified_routing": BenchmarkMode(
        name="no_orchestrator_simplified_routing",
        description="Ablation that bypasses the orchestrator router and uses simplified keyword routing.",
        route_strategy="rule_based",
    ),
    "single_agent_gpt4o_placeholder": BenchmarkMode(
        name="single_agent_gpt4o_placeholder",
        description="Interface placeholder for a future approved live single-agent GPT-4o baseline.",
        supported_routes=(),
        placeholder=True,
    ),
    "single_agent_rag_placeholder": BenchmarkMode(
        name="single_agent_rag_placeholder",
        description="Interface placeholder for a future approved live single-agent RAG baseline.",
        supported_routes=(),
        placeholder=True,
    ),
}


def load_benchmark_prompts(path: str | Path) -> list[BenchmarkPrompt]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Benchmark prompts file must contain a JSON list")
    return [BenchmarkPrompt.from_mapping(item) for item in raw]


def load_gold_benchmark_prompts(path: str | Path) -> list[BenchmarkPrompt]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Gold benchmark prompts file must contain a JSON list")
    return [BenchmarkPrompt.from_mapping(item, require_gold_fields=True) for item in raw]


def run_mock_benchmark(
    *,
    prompts_path: str | Path = DEFAULT_PROMPTS_PATH,
    reviews_path: str | Path = DEFAULT_REVIEWS_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    run_id: str | None = None,
    dataset_name: str = "mock_benchmark_reviews_fixture",
    product_name: str = "sample product",
    benchmark_modes: Sequence[str] | None = None,
    require_gold_schema: bool = False,
    evidence_id: str = EVIDENCE_ID,
) -> BenchmarkArtifacts:
    prompts_path = Path(prompts_path)
    reviews_path = Path(reviews_path)
    output_root = Path(output_dir)
    run_id = run_id or f"mock_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    prompts = load_gold_benchmark_prompts(prompts_path) if require_gold_schema else load_benchmark_prompts(prompts_path)
    review_rows = json.loads(reviews_path.read_text(encoding="utf-8"))
    selected_modes = _resolve_modes(benchmark_modes)
    provider = MockLLMProvider()
    results: list[BenchmarkResult] = []

    for mode_spec in selected_modes:
        settings = load_settings(
            {
                "LLM_REVIEW_PROJECT_ROOT": str(run_dir),
                "REVIEWS_DB_PATH": str(run_dir / "runtime" / f"{mode_spec.name}.db"),
                "OUTPUT_DIR": str(run_dir / "charts" / mode_spec.name),
                "VECTORSTORE_DIR": str(run_dir / "vectorstores" / mode_spec.name),
                "LLM_PROVIDER": "mock",
                "LLM_MODEL": "mock-llm",
                "SEMANTIC_RETRIEVAL_BACKEND": "lexical",
                "ALLOW_LIVE_LLM": "false",
                "ALLOW_LIVE_RETRIEVAL": "false",
            }
        )
        ensure_directories(settings)
        mode_rows = _apply_mode_ablation(review_rows, mode_spec)
        with sqlite3.connect(settings.database_path) as conn:
            conn.row_factory = sqlite3.Row
            RetrievalAgent(settings).load_records(conn, product_name, mode_rows)
            orchestrator = ReviewOrchestrator(settings, provider)
            results.extend(
                _run_prompt(run_id, mode_spec, orchestrator, conn, prompt, provider.model)
                for prompt in prompts
            )

    manifest = BenchmarkRunManifest(
        run_id=run_id,
        mode="mock",
        modes=tuple(mode.name for mode in selected_modes),
        dataset_name=dataset_name,
        prompts_path=str(prompts_path),
        reviews_path=str(reviews_path),
        output_dir=str(run_dir),
        prompt_count=len(prompts),
        result_count=len(results),
        live_mode=False,
        model_provider="MockLLMProvider",
        model=provider.model,
        command=" ".join(sys.argv),
        evaluation_tags=BENCHMARK_EVALUATION_TAGS,
        gold_schema_required=require_gold_schema,
    )
    metrics = summarize_benchmark_results(results)
    limitations = [
        "Mock-mode benchmark harness validation only.",
        "Does not validate live GPT-4o behavior, live LangChain/FAISS retrieval, real API costs, scalability, or broad empirical superiority.",
        "Placeholder single-agent GPT-4o and RAG modes are intentionally not executed before live-run approval.",
    ]
    summary = {
        "run_id": run_id,
        "mode": "mock",
        "modes": [mode.name for mode in selected_modes],
        "dataset_name": dataset_name,
        "evaluation_tags": list(BENCHMARK_EVALUATION_TAGS),
        "gold_schema_required": require_gold_schema,
        "metrics": metrics,
        "output_files": {},
        "limitations": limitations,
    }

    manifest_path = run_dir / "manifest.json"
    results_path = run_dir / "results.jsonl"
    summary_path = run_dir / "summary.json"
    evidence_path = run_dir / "evidence.json"
    summary["output_files"] = {
        "manifest": str(manifest_path),
        "results": str(results_path),
        "summary": str(summary_path),
        "evidence": str(evidence_path),
    }
    evidence = {
        "evidence_id": evidence_id,
        "run_id": run_id,
        "date_time_utc": datetime.now(timezone.utc).isoformat(),
        "live_mock_status": "mock/offline; no live API calls",
        "model_provider": "MockLLMProvider",
        "input_data": {
            "prompts_path": str(prompts_path),
            "reviews_path": str(reviews_path),
            "dataset_name": dataset_name,
        },
        "modes": [mode.name for mode in selected_modes],
        "gold_schema_required": require_gold_schema,
        "prompt_count": len(prompts),
        "result_count": len(results),
        "output_files": summary["output_files"],
        "key_results": summary["metrics"],
        "evaluation_tags": list(BENCHMARK_EVALUATION_TAGS),
        "limitations": limitations,
    }

    _write_json(manifest_path, manifest.to_dict())
    _write_jsonl(results_path, [result.to_dict() for result in results])
    _write_json(summary_path, summary)
    _write_json(evidence_path, evidence)
    return BenchmarkArtifacts(
        run_id=run_id,
        run_dir=run_dir,
        manifest_path=manifest_path,
        results_path=results_path,
        summary_path=summary_path,
        evidence_path=evidence_path,
    )


def _run_prompt(
    run_id: str,
    mode_spec: BenchmarkMode,
    orchestrator: ReviewOrchestrator,
    conn: sqlite3.Connection,
    prompt: BenchmarkPrompt,
    model: str,
) -> BenchmarkResult:
    start = time.perf_counter()
    checks: list[BenchmarkCheck] = []
    trace: dict[str, Any] = {}
    result: dict[str, Any] = {}
    failure_category = None
    failure_reason = None
    actual_sql = None
    sql_valid = None
    sql_execution_status = "not_applicable"

    if mode_spec.placeholder:
        latency_ms = round((time.perf_counter() - start) * 1000.0, 3)
        checks.append(BenchmarkCheck("infrastructure_recorded", True, True, True))
        checks.append(
            BenchmarkCheck(
                name="mode_executable",
                passed=False,
                expected="mock execution",
                actual="placeholder only",
                details=mode_spec.description,
            )
        )
        return _build_result(
            run_id=run_id,
            mode_spec=mode_spec,
            prompt=prompt,
            model=model,
            result={"type": "placeholder", "message": mode_spec.description},
            trace={"route": None, "table": None, "product_name": None, "date_range": None},
            checks=checks,
            latency_ms=latency_ms,
            actual_sql=None,
            sql_valid=None,
            sql_execution_status="not_executed",
            failure_category="placeholder_not_executed",
            failure_reason=mode_spec.description,
        )

    inapplicable_reason = _mode_inapplicable_reason(mode_spec, prompt)
    if inapplicable_reason:
        latency_ms = round((time.perf_counter() - start) * 1000.0, 3)
        checks.append(BenchmarkCheck("infrastructure_recorded", True, True, True))
        checks.append(
            BenchmarkCheck(
                "mode_applicable",
                False,
                "prompt executable by this mode",
                "inapplicable",
                inapplicable_reason,
            )
        )
        return _build_result(
            run_id=run_id,
            mode_spec=mode_spec,
            prompt=prompt,
            model=model,
            result={"type": "inapplicable", "message": inapplicable_reason},
            trace={"route": None, "table": None, "product_name": None, "date_range": None},
            checks=checks,
            latency_ms=latency_ms,
            actual_sql=None,
            sql_valid=None,
            sql_execution_status="not_applicable",
            failure_category="inapplicable_for_mode",
            failure_reason=inapplicable_reason,
            success=False,
            inapplicable_reason=inapplicable_reason,
        )

    try:
        route = _select_route(mode_spec, orchestrator, prompt.prompt_text)
        if route not in mode_spec.supported_routes:
            trace = _base_trace(orchestrator, conn, prompt.prompt_text, route)
            result = {
                "type": "error",
                "message": f"Mode {mode_spec.name} does not support route {route}.",
            }
            failure_category = "unsupported_route_for_mode"
            failure_reason = result["message"]
        else:
            result, trace = _execute_route(orchestrator, conn, prompt.prompt_text, route)
        controlled_category = _optional_str(trace.get("failure_category") or result.get("failure_category"))
        if controlled_category:
            failure_category = controlled_category
            failure_reason = _optional_str(trace.get("failure_reason") or result.get("failure_reason") or result.get("message"))
        actual_sql = _optional_str(trace.get("sql"))
        sql_valid, sql_execution_status = _sql_status(actual_sql, trace.get("table"), trace, failure_category)
    except Exception as exc:  # pragma: no cover - failure rows are primarily driven by fixtures.
        failure_category = type(exc).__name__
        failure_reason = _redact_sensitive_text(str(exc))
        result = {"type": "error", "message": failure_reason}
        trace = {"route": None, "table": None, "product_name": None, "date_range": None}
        sql_execution_status = "error"

    latency_ms = round((time.perf_counter() - start) * 1000.0, 3)
    checks.extend(_evaluate_checks(prompt, result, trace, actual_sql, sql_valid, sql_execution_status))
    expected_failure_was_handled: bool | None = None
    if prompt.expected_failure_category:
        expected_failure_was_handled = failure_category == prompt.expected_failure_category
        if not expected_failure_was_handled and failure_category is None:
            failure_category = f"expected_failure_not_handled_{prompt.expected_failure_category}"
            failure_reason = f"Expected controlled failure was not recorded: {prompt.expected_failure_category}"
        checks.append(
            BenchmarkCheck(
                "expected_failure_handled",
                expected_failure_was_handled,
                prompt.expected_failure_category,
                failure_category,
            )
        )
    success = _result_success(checks, failure_category, expected_failure_was_handled)
    if not success and failure_category is None:
        failure_category = _first_failed_check_category(checks)
        failure_reason = _failure_reason(checks)

    return _build_result(
        run_id=run_id,
        mode_spec=mode_spec,
        prompt=prompt,
        model=model,
        result=result,
        trace=trace,
        checks=checks,
        latency_ms=latency_ms,
        actual_sql=actual_sql,
        sql_valid=sql_valid,
        sql_execution_status=sql_execution_status,
        failure_category=failure_category,
        failure_reason=failure_reason,
        success=success,
    )


def _mode_inapplicable_reason(mode_spec: BenchmarkMode, prompt: BenchmarkPrompt) -> str | None:
    if prompt.expected_route in mode_spec.supported_routes:
        return None
    supported = ", ".join(mode_spec.supported_routes) or "no executable routes"
    return (
        f"Mode {mode_spec.name} supports {supported}, "
        f"but this gold item expects {prompt.expected_route}."
    )


def _execute_route(
    orchestrator: ReviewOrchestrator,
    conn: sqlite3.Connection,
    prompt_text: str,
    route: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    trace = _base_trace(orchestrator, conn, prompt_text, route)
    table = trace.get("table")
    if not table:
        target = trace.get("product_name") or "the requested product"
        trace["failure_category"] = "product_not_found"
        trace["failure_reason"] = "No review table matched the requested product/table name."
        return {
            "type": "text",
            "message": f"No matching data found for {target}. The available tables contain 0 reviews for that requested product.",
            "failure_category": "product_not_found",
            "failure_reason": trace["failure_reason"],
        }, trace
    if route == "ANALYTICS":
        result = orchestrator.analytics_agent.run(conn, table, prompt_text)
        if result.get("failure_category"):
            trace["failure_category"] = str(result.get("failure_category"))
            trace["failure_reason"] = str(result.get("failure_reason") or result.get("message") or "")
        trace.update(
            {
                "chart_path": result.get("path"),
                "chart_type": result.get("chart_type"),
                "chart_group_by": result.get("group_by"),
                "chart_aggregation": result.get("aggregation"),
                "chart_rows": result.get("chart_rows", []),
            }
        )
        return result, trace
    if route == "SEMANTICS":
        metadata = orchestrator.extract_metadata(prompt_text)
        controlled_failure = orchestrator._semantic_controlled_failure(conn, table, prompt_text, metadata.product_name)
        if controlled_failure:
            trace["failure_category"] = controlled_failure.category
            trace["failure_reason"] = controlled_failure.reason
            return {
                "type": "text",
                "message": controlled_failure.message,
                "failure_category": controlled_failure.category,
                "failure_reason": controlled_failure.reason,
            }, trace
        semantic_trace = orchestrator.semantic_reasoning_agent.answer_with_trace(conn, table, prompt_text)
        trace["evidence_ids"] = list(semantic_trace.evidence_ids)
        trace["evidence_snippets"] = list(semantic_trace.evidence_snippets)
        return {"type": "text", "message": semantic_trace.answer}, trace
    message, direct_trace = orchestrator._run_direct_sql(conn, table, prompt_text)
    trace["sql"] = direct_trace.sql
    trace["sql_columns"] = list(direct_trace.columns)
    trace["sql_row_count"] = direct_trace.row_count
    trace["sql_planner"] = direct_trace.planner
    trace["sql_execution_status"] = "executed"
    return {"type": "text", "message": message}, trace


def _base_trace(orchestrator: ReviewOrchestrator, conn: sqlite3.Connection, prompt_text: str, route: str | None) -> dict[str, Any]:
    metadata = orchestrator.extract_metadata(prompt_text)
    table = orchestrator._match_table(conn, metadata.product_name)
    product_name = orchestrator.display_product_name(metadata.product_name, table)
    return {
        "prompt": prompt_text,
        "product_name": product_name,
        "date_range": metadata.date_range,
        "table": table,
        "route": route,
        "sql": None,
        "sql_execution_status": "not_applicable",
        "evidence_ids": [],
        "evidence_snippets": [],
        "chart_path": None,
        "chart_type": None,
        "chart_rows": [],
        "failure_category": None,
        "failure_reason": None,
    }


def _select_route(mode_spec: BenchmarkMode, orchestrator: ReviewOrchestrator, prompt_text: str) -> str:
    if mode_spec.route_strategy == "sql_only":
        return "DIRECT_SQL"
    if mode_spec.route_strategy == "rule_based":
        return _rule_based_route(prompt_text)
    return orchestrator.route(prompt_text)


def _rule_based_route(prompt_text: str) -> str:
    lower = prompt_text.lower()
    if any(word in lower for word in ("plot", "chart", "visual", "distribution", "trend", "show")):
        return "ANALYTICS"
    if any(phrase in lower for phrase in ("how many", "count", "number of", "average", "avg")):
        return "DIRECT_SQL"
    if any(word in lower for word in ("why", "how", "problem", "issue", "good", "bad", "broken", "misleading", "contradictory", "mixed", "por que")):
        return "SEMANTICS"
    return "DIRECT_SQL"


def _evaluate_checks(
    prompt: BenchmarkPrompt,
    result: dict[str, Any],
    trace: dict[str, Any],
    actual_sql: str | None,
    sql_valid: bool | None,
    sql_execution_status: str,
) -> list[BenchmarkCheck]:
    checks: list[BenchmarkCheck] = []
    actual_route = _optional_str(trace.get("route"))
    actual_result_type = _optional_str(result.get("type"))
    response_text = _response_text(result)
    actual_table = _optional_str(trace.get("table"))
    actual_product = _optional_str(trace.get("product_name"))
    actual_date_range = _optional_str(trace.get("date_range"))

    checks.append(BenchmarkCheck("infrastructure_recorded", True, True, True))
    checks.append(BenchmarkCheck("route", actual_route == prompt.expected_route, prompt.expected_route, actual_route))
    if prompt.expected_result_type:
        checks.append(
            BenchmarkCheck(
                "result_type",
                actual_result_type == prompt.expected_result_type,
                prompt.expected_result_type,
                actual_result_type,
            )
        )
    if prompt.expected_table or prompt.expected_product_name:
        expected = {"product": prompt.expected_product_name, "table": prompt.expected_table}
        actual = {"product": actual_product, "table": actual_table}
        checks.append(
            BenchmarkCheck(
                "product_table",
                _product_table_matches(prompt, actual_product, actual_table),
                expected,
                actual,
            )
        )
    if prompt.expected_date_range:
        checks.append(
            BenchmarkCheck(
                "date_range",
                actual_date_range == prompt.expected_date_range,
                prompt.expected_date_range,
                actual_date_range,
            )
        )
    if actual_sql or prompt.expected_sql or prompt.expected_sql_pattern or prompt.expected_route == "DIRECT_SQL":
        checks.append(BenchmarkCheck("sql_valid", sql_valid is True, True, sql_valid))
        checks.append(BenchmarkCheck("sql_execution", sql_execution_status == "executed", "executed", sql_execution_status))
    if prompt.expected_sql and not prompt.expected_sql_pattern:
        checks.append(
            BenchmarkCheck(
                "sql_exact",
                normalize_sql(actual_sql) == normalize_sql(prompt.expected_sql),
                prompt.expected_sql,
                actual_sql,
            )
        )
    if prompt.expected_sql_pattern:
        checks.append(
            BenchmarkCheck(
                "sql_pattern",
                bool(actual_sql and re.search(prompt.expected_sql_pattern, actual_sql, flags=re.IGNORECASE)),
                prompt.expected_sql_pattern,
                actual_sql,
            )
        )
    if prompt.expected_answer_contains:
        checks.append(
            BenchmarkCheck(
                "answer_contains",
                contains_all(response_text, prompt.expected_answer_contains),
                list(prompt.expected_answer_contains),
                response_text,
            )
        )
    if prompt.expected_answer_facts:
        checks.append(
            BenchmarkCheck(
                "answer_facts",
                contains_all(response_text, prompt.expected_answer_facts),
                list(prompt.expected_answer_facts),
                response_text,
            )
        )
    evidence_snippets = tuple(str(value) for value in trace.get("evidence_snippets", ()))
    evidence_ids = tuple(str(value) for value in trace.get("evidence_ids", ()))
    if prompt.expected_source_review_ids:
        expected_ids = set(prompt.expected_source_review_ids)
        actual_ids = set(evidence_ids)
        checks.append(
            BenchmarkCheck(
                "source_review_ids",
                expected_ids.issubset(actual_ids),
                sorted(expected_ids),
                sorted(actual_ids),
            )
        )
    if prompt.expected_evidence_snippets:
        evidence_text = " ".join(evidence_snippets) + " " + response_text
        checks.append(
            BenchmarkCheck(
                "evidence_snippets",
                contains_all(evidence_text, prompt.expected_evidence_snippets),
                list(prompt.expected_evidence_snippets),
                evidence_text,
            )
        )
    if prompt.expected_evidence_contains:
        evidence_text = " ".join(evidence_snippets) + " " + response_text
        checks.append(
            BenchmarkCheck(
                "evidence_contains",
                contains_all(evidence_text, prompt.expected_evidence_contains),
                list(prompt.expected_evidence_contains),
                evidence_text,
            )
        )
    if prompt.expected_chart_type:
        checks.append(
            BenchmarkCheck(
                "chart_type",
                result.get("chart_type") == prompt.expected_chart_type,
                prompt.expected_chart_type,
                result.get("chart_type"),
            )
        )
    if prompt.expected_chart_group_by:
        checks.append(
            BenchmarkCheck(
                "chart_group_by",
                result.get("group_by") == prompt.expected_chart_group_by,
                prompt.expected_chart_group_by,
                result.get("group_by"),
            )
        )
    chart_path = _optional_str(result.get("path"))
    if actual_result_type == "chart" or prompt.expected_chart_type:
        checks.append(
            BenchmarkCheck(
                "chart_file_exists",
                bool(chart_path and Path(chart_path).exists()),
                True,
                bool(chart_path and Path(chart_path).exists()),
                chart_path or "",
            )
        )
    if prompt.expected_numeric_values:
        actual_numeric_values = _chart_numeric_values(trace, result)
        checks.append(
            BenchmarkCheck(
                "chart_numeric_values",
                _numeric_values_match(prompt.expected_numeric_values, actual_numeric_values),
                prompt.expected_numeric_values,
                actual_numeric_values,
            )
        )
    return checks


def _build_result(
    *,
    run_id: str,
    mode_spec: BenchmarkMode,
    prompt: BenchmarkPrompt,
    model: str,
    result: dict[str, Any],
    trace: dict[str, Any],
    checks: list[BenchmarkCheck],
    latency_ms: float,
    actual_sql: str | None,
    sql_valid: bool | None,
    sql_execution_status: str,
    failure_category: str | None,
    failure_reason: str | None,
    success: bool | None = None,
    inapplicable_reason: str | None = None,
) -> BenchmarkResult:
    evidence_snippets = tuple(str(value) for value in trace.get("evidence_snippets", ()))
    evidence_containment_status = _combined_status(checks, ("evidence_contains", "evidence_snippets", "source_review_ids"))
    chart_numeric_values = _chart_numeric_values(trace, result)
    chart_numerical_consistency = _check_status(checks, "chart_numeric_values")
    actual_chart_type = _optional_str(result.get("chart_type"))
    infrastructure_success = _check_status(checks, "infrastructure_recorded") is not False
    route_correctness = _check_status(checks, "route")
    factual_correctness_proxy = _combined_status(checks, ("answer_facts", "answer_contains"))
    chart_structural_correctness = _combined_status(checks, ("chart_type", "chart_group_by", "chart_file_exists"))
    expected_failure_handled = _check_status(checks, "expected_failure_handled")
    return BenchmarkResult(
        run_id=run_id,
        prompt_id=prompt.prompt_id,
        prompt_text=prompt.prompt_text,
        category=prompt.category,
        language=prompt.language,
        mode=mode_spec.name,
        mode_name=mode_spec.name,
        model_provider="MockLLMProvider",
        model=model,
        expected_route=prompt.expected_route,
        actual_route=_optional_str(trace.get("route")),
        expected_product=prompt.expected_product_name,
        actual_product=_optional_str(trace.get("product_name")),
        expected_table=prompt.expected_table,
        actual_table=_optional_str(trace.get("table")),
        expected_date_range=prompt.expected_date_range,
        actual_date_range=_optional_str(trace.get("date_range")),
        expected_sql=prompt.expected_sql,
        expected_sql_pattern=prompt.expected_sql_pattern,
        actual_sql=actual_sql,
        sql_valid=sql_valid,
        sql_execution_status=sql_execution_status,
        expected_result_type=prompt.expected_result_type,
        actual_result_type=_optional_str(result.get("type")),
        expected_answer_facts=prompt.expected_answer_facts,
        expected_source_review_ids=prompt.expected_source_review_ids,
        expected_evidence_snippets=prompt.expected_evidence_snippets,
        expected_evidence=prompt.expected_evidence_contains,
        evidence_containment_status=evidence_containment_status,
        expected_chart_type=prompt.expected_chart_type,
        actual_chart_type=actual_chart_type,
        expected_chart_grouping=prompt.expected_chart_grouping,
        expected_chart_values=prompt.expected_chart_values,
        actual_chart_values=chart_numeric_values,
        expected_numeric_values=prompt.expected_numeric_values,
        actual_numeric_values=chart_numeric_values,
        chart_numerical_consistency=chart_numerical_consistency,
        expected_failure_type=prompt.expected_failure_type,
        expected_refusal_category=prompt.expected_refusal_category,
        sql_eligible=prompt.sql_eligible,
        sql_execution_eligible=prompt.sql_execution_eligible,
        chart_generation_eligible=prompt.chart_generation_eligible,
        chart_numeric_eligible=prompt.chart_numeric_eligible,
        evidence_required=prompt.evidence_required,
        answer_fact_eligible=prompt.answer_fact_eligible,
        success_criteria=prompt.success_criteria,
        gold_verification_status=prompt.gold_verification_status,
        gold_verified_by=prompt.gold_verified_by,
        gold_computation_method=prompt.gold_computation_method,
        gold_source_query=prompt.gold_source_query,
        gold_source_records=prompt.gold_source_records,
        gold_notes=prompt.gold_notes,
        infrastructure_success=infrastructure_success,
        route_correctness=route_correctness,
        factual_correctness_proxy=factual_correctness_proxy,
        evidence_containment=evidence_containment_status,
        chart_structural_correctness=chart_structural_correctness,
        chart_numerical_correctness=chart_numerical_consistency,
        expected_failure_handled=expected_failure_handled,
        success=_result_success(checks, failure_category, expected_failure_handled) if success is None else success,
        failure_category=failure_category,
        failure_reason=failure_reason,
        latency_ms=latency_ms,
        checks=tuple(checks),
        chart_path=_optional_str(result.get("path")),
        chart_group_by=_optional_str(result.get("group_by")),
        evidence_ids=tuple(str(value) for value in trace.get("evidence_ids", ())),
        evidence_snippets=evidence_snippets,
        response_preview=_response_text(result)[:500],
        input_tokens=None,
        output_tokens=None,
        total_tokens=None,
        estimated_cost_usd=None,
        token_usage=None,
        failure_type=failure_category,
        failure_message=failure_reason,
        mode_execution_type="mock",
        uses_live_gpt4o=False,
        uses_faiss=False,
        uses_sql="DIRECT_SQL" in mode_spec.supported_routes,
        uses_mock_provider=True,
        uses_deterministic_logic=mode_spec.route_strategy in {"rule_based", "sql_only"} or mode_spec.placeholder,
        disabled_enrichment_fields=mode_spec.disabled_fields,
        inapplicable_reason=inapplicable_reason,
        live_call_count=0,
        deterministic_call_count=1 if mode_spec.route_strategy in {"rule_based", "sql_only"} or inapplicable_reason else 0,
        mock_call_count=1,
    )


def _sql_status(
    actual_sql: str | None,
    actual_table: Any,
    trace: dict[str, Any],
    failure_category: str | None,
) -> tuple[bool | None, str]:
    if not actual_sql:
        if failure_category:
            return None, "not_executed"
        return None, "not_applicable"
    table = _optional_str(actual_table)
    if not table:
        return False, "invalid"
    try:
        validate_select_sql(actual_sql, allowed_tables=[table], allowed_columns=REVIEW_COLUMNS)
    except SQLValidationError:
        return False, "invalid"
    return True, str(trace.get("sql_execution_status") or "executed")


def _apply_mode_ablation(rows: Iterable[dict[str, Any]], mode_spec: BenchmarkMode) -> list[dict[str, Any]]:
    transformed: list[dict[str, Any]] = []
    for row in rows:
        copy = dict(row)
        for field in mode_spec.disabled_fields:
            copy[field] = ""
        transformed.append(copy)
    return transformed


def _resolve_modes(benchmark_modes: Sequence[str] | None) -> list[BenchmarkMode]:
    if not benchmark_modes:
        return list(BENCHMARK_MODES.values())
    names = []
    for value in benchmark_modes:
        names.extend(part.strip() for part in str(value).split(",") if part.strip())
    if names == ["all"]:
        return list(BENCHMARK_MODES.values())
    unknown = sorted(name for name in names if name not in BENCHMARK_MODES)
    if unknown:
        raise ValueError(f"Unsupported benchmark mode(s): {', '.join(unknown)}")
    return [BENCHMARK_MODES[name] for name in names]


def _product_table_matches(prompt: BenchmarkPrompt, actual_product: str | None, actual_table: str | None) -> bool:
    if prompt.expected_table and actual_table != prompt.expected_table:
        return False
    if prompt.expected_product_name:
        try:
            return normalize_table_name(actual_product or "") == normalize_table_name(prompt.expected_product_name)
        except ValueError:
            return False
    return True


def _chart_numeric_values(trace: dict[str, Any], result: dict[str, Any]) -> dict[str, float]:
    rows = trace.get("chart_rows") or result.get("chart_rows") or []
    values: dict[str, float] = {}
    for row in rows:
        if isinstance(row, dict) and "label" in row and "value" in row:
            values[str(row["label"])] = float(row["value"])
        elif isinstance(row, (list, tuple)) and len(row) >= 2:
            values[str(row[0])] = float(row[1])
    return values


def _numeric_values_match(expected: dict[str, float], actual: dict[str, float], tolerance: float = 1e-6) -> bool:
    if set(expected) != set(actual):
        return False
    return all(abs(float(expected[key]) - float(actual[key])) <= tolerance for key in expected)


def _check_status(checks: list[BenchmarkCheck], name: str) -> bool | None:
    for check in checks:
        if check.name == name:
            return check.passed
    return None


def _combined_status(checks: list[BenchmarkCheck], names: tuple[str, ...]) -> bool | None:
    statuses = [check.passed for check in checks if check.name in names]
    if not statuses:
        return None
    return all(statuses)


def _result_success(
    checks: list[BenchmarkCheck],
    failure_category: str | None,
    expected_failure_handled: bool | None,
) -> bool:
    if expected_failure_handled is True:
        return _check_status(checks, "infrastructure_recorded") is not False
    if expected_failure_handled is False:
        return False
    return failure_category is None and all(check.passed for check in checks)


def _first_failed_check_category(checks: list[BenchmarkCheck]) -> str | None:
    for check in checks:
        if not check.passed:
            return f"failed_{check.name}"
    return None


def _failure_reason(checks: list[BenchmarkCheck]) -> str | None:
    failed = [check.name for check in checks if not check.passed]
    if not failed:
        return None
    return "Failed check(s): " + ", ".join(failed)


def _response_text(result: dict[str, Any]) -> str:
    return str(result.get("message") or result.get("explanation") or "")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _redact_sensitive_text(text: str) -> str:
    redacted = str(text)
    for key in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "HF_TOKEN",
        "HUGGINGFACEHUB_API_TOKEN",
        "HF_ENDPOINT_URL",
        "HF_LLAMA_ENDPOINT_URL",
        "HF_QWEN_ENDPOINT_URL",
    ):
        value = os.environ.get(key)
        if value:
            label = "<redacted endpoint URL>" if key.endswith("ENDPOINT_URL") else "<redacted secret>"
            redacted = redacted.replace(value, label)
    redacted = re.sub(
        r"https://[A-Za-z0-9._/-]*endpoints\.huggingface\.cloud(?:/[^\s'\"}]*)?",
        "<redacted endpoint URL>",
        redacted,
    )
    redacted = re.sub(r"\b(?:hf|sk|sk-ant)-[A-Za-z0-9_\-]{20,}\b", "<redacted secret>", redacted)
    return redacted


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the mock benchmark harness and write manifest/results/evidence artifacts.")
    parser.add_argument("--mode", default="mock", choices=("mock",), help="Only mock mode is enabled before live-run approval.")
    parser.add_argument("--benchmark-modes", nargs="*", default=("all",), help="Benchmark mode names, comma-separated names, or all.")
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS_PATH)
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dataset-name", default="mock_benchmark_reviews_fixture")
    parser.add_argument("--product-name", default="sample product")
    parser.add_argument("--require-gold-schema", action="store_true", help="Require verified gold benchmark fields.")
    parser.add_argument("--evidence-id", default=EVIDENCE_ID)
    args = parser.parse_args()

    artifacts = run_mock_benchmark(
        prompts_path=args.prompts,
        reviews_path=args.reviews,
        output_dir=args.output_dir,
        run_id=args.run_id,
        dataset_name=args.dataset_name,
        product_name=args.product_name,
        benchmark_modes=args.benchmark_modes,
        require_gold_schema=args.require_gold_schema,
        evidence_id=args.evidence_id,
    )
    print(f"run_id={artifacts.run_id}")
    print(f"manifest={artifacts.manifest_path}")
    print(f"results={artifacts.results_path}")
    print(f"summary={artifacts.summary_path}")
    print(f"evidence={artifacts.evidence_path}")


if __name__ == "__main__":
    main()

