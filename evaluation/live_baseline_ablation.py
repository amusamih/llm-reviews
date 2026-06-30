from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import sys
from typing import Any, Mapping, Sequence
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (SRC_ROOT, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from evaluation.benchmark_metrics import latency_summary, summarize_benchmark_results
from evaluation.benchmark_schema import BenchmarkArtifacts, BenchmarkPrompt, BenchmarkResult
from evaluation.failure_examples import collect_failure_examples
from evaluation.live_pilot import (
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROVIDER,
    DEFAULT_REVIEWS_PATH,
    UsageTrackingProvider,
    _git_commit_sha,
    _install_process_environment,
    _prompt_had_api_failure,
    _scrubbed_provider_config,
    _token_summary,
    _with_live_usage,
    read_dotenv,
)
from evaluation.run_benchmark import (
    BENCHMARK_MODES,
    BENCHMARK_EVALUATION_TAGS,
    BenchmarkMode,
    _apply_mode_ablation,
    _run_prompt,
    _write_json,
    _write_jsonl,
    load_gold_benchmark_prompts,
)
from llm_review_analysis.agents import RetrievalAgent, ReviewOrchestrator
from llm_review_analysis.config import ensure_directories, load_settings
from llm_review_analysis.providers import build_llm_provider


DEFAULT_EVIDENCE_ID = "EVID-LIVE-BASELINE-ABLATION-001"
DEFAULT_RUN_ID_PREFIX = "live_baseline_ablation"
DEFAULT_PROMPTS_PATH = PROJECT_ROOT / "evaluation" / "alternative_workflow_prompts.json"
DEFAULT_MODE_NAMES = (
    "proposed_multi_agent_live_gpt4o",
    "rule_based_routing_baseline",
    "sql_only",
    "sql_plus_semantic_retrieval",
    "no_semantic_tags",
    "no_topic_or_aspect_tags",
    "no_translation",
    "no_vector_retrieval",
    "no_orchestrator_simplified_routing",
)


@dataclass(frozen=True)
class LiveModeSpec:
    mode: BenchmarkMode
    mode_execution_type: str
    uses_live_gpt4o: bool
    uses_faiss: bool
    uses_sql: bool
    uses_mock_provider: bool
    uses_deterministic_logic: bool
    semantic_backend: str
    disabled_enrichment_fields: tuple[str, ...] = ()
    notes: str = ""


def resolve_live_modes(mode_names: Sequence[str] | None = None) -> list[LiveModeSpec]:
    names = _split_mode_names(mode_names) or list(DEFAULT_MODE_NAMES)
    if names == ["all"]:
        names = list(DEFAULT_MODE_NAMES)
    unknown = sorted(name for name in names if name not in LIVE_MODE_SPECS)
    if unknown:
        raise ValueError(f"Unsupported live baseline/ablation mode(s): {', '.join(unknown)}")
    return [LIVE_MODE_SPECS[name] for name in names]


def run_live_baseline_ablation(
    *,
    prompts_path: str | Path = DEFAULT_PROMPTS_PATH,
    reviews_path: str | Path = DEFAULT_REVIEWS_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    run_id: str | None = None,
    dataset_name: str = "amazon_all_beauty_programmatic_gold_live_baseline_ablation",
    product_name: str = "amazon all beauty",
    provider_name: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
    max_prompts: int = 11,
    max_api_failures: int = 3,
    evidence_id: str = DEFAULT_EVIDENCE_ID,
    mode_names: Sequence[str] | None = None,
) -> BenchmarkArtifacts:
    prompts_path = Path(prompts_path)
    reviews_path = Path(reviews_path)
    output_root = Path(output_dir)
    run_id = run_id or f"{DEFAULT_RUN_ID_PREFIX}_{datetime.now(timezone.utc).strftime('%Y%m%d')}_{uuid4().hex[:4]}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    selected_modes = resolve_live_modes(mode_names)
    prompts = [
        prompt
        for prompt in load_gold_benchmark_prompts(prompts_path)
        if prompt.gold_verification_status == "programmatically_verified"
    ][:max_prompts]
    review_rows = json.loads(reviews_path.read_text(encoding="utf-8"))
    dotenv_values = read_dotenv(PROJECT_ROOT / ".env")
    if not dotenv_values.get("OPENAI_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not available for the approved live baseline/ablation phase.")

    estimated_calls = estimate_live_operations(prompts, selected_modes)
    results: list[BenchmarkResult] = []
    api_failure_count = 0
    stopped_early_reason: str | None = None

    for live_spec in selected_modes:
        mode_spec = live_spec.mode
        mode_env = dict(dotenv_values)
        mode_env.update(
            {
                "LLM_REVIEW_PROJECT_ROOT": str(run_dir),
                "REVIEWS_DB_PATH": str(run_dir / "runtime" / f"{mode_spec.name}.db"),
                "OUTPUT_DIR": str(run_dir / "charts" / mode_spec.name),
                "VECTORSTORE_DIR": str(run_dir / "vectorstores" / mode_spec.name),
                "LLM_PROVIDER": provider_name,
                "LLM_MODEL": model,
                "SEMANTIC_RETRIEVAL_BACKEND": live_spec.semantic_backend,
                "ALLOW_LIVE_LLM": "true",
                "ALLOW_LIVE_RETRIEVAL": "false",
            }
        )
        _install_process_environment(mode_env)
        settings = load_settings(mode_env)
        ensure_directories(settings)
        wrapped_provider = build_llm_provider(settings)
        provider = UsageTrackingProvider(
            wrapped_provider,
            provider_name=type(wrapped_provider).__name__,
            model=getattr(wrapped_provider, "model", model),
        )
        mode_rows = _apply_mode_ablation(review_rows, mode_spec)
        with sqlite3.connect(settings.database_path) as conn:
            conn.row_factory = sqlite3.Row
            RetrievalAgent(settings).load_records(conn, product_name, mode_rows)
            orchestrator = ReviewOrchestrator(settings, provider)
            for prompt in prompts:
                provider.start_prompt(f"{mode_spec.name}:{prompt.prompt_id}")
                result = _run_prompt(run_id, mode_spec, orchestrator, conn, prompt, provider.model)
                usage = provider.finish_prompt()
                result = _with_live_usage(result, provider, usage)
                result = _with_mode_metadata(result, live_spec)
                results.append(result)
                if _prompt_had_api_failure(usage):
                    api_failure_count += 1
                if api_failure_count >= max_api_failures:
                    stopped_early_reason = f"Stopped after {api_failure_count} provider/API failure(s)."
                    break
        if stopped_early_reason:
            break

    metrics = summarize_benchmark_results(results)
    mode_comparison = build_mode_comparison(metrics, results, selected_modes)
    token_summary = _token_summary(results)
    category_counts = dict(sorted(Counter(result.category for result in results).items()))
    output_files: dict[str, str] = {}
    manifest = {
        "run_id": run_id,
        "mode": "live_baseline_ablation",
        "modes": [live_spec.mode.name for live_spec in selected_modes],
        "mode_descriptions": _mode_description_payload(selected_modes),
        "dataset_name": dataset_name,
        "prompts_path": str(prompts_path),
        "reviews_path": str(reviews_path),
        "output_dir": str(run_dir),
        "prompt_count": len(prompts),
        "executed_result_count": len(results),
        "live_mode": True,
        "model_provider": provider_name,
        "model": model,
        "provider_config": _scrubbed_provider_config(dotenv_values),
        "command": " ".join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "code_version": _git_commit_sha(),
        "evaluation_tags": list(BENCHMARK_EVALUATION_TAGS),
        "gold_schema_required": True,
        "gold_inclusion_rule": "programmatically_verified prompts only; author/human-unverified judgment prompts excluded",
        "cost_control": {
            "pre_run_estimated_operations": estimated_calls,
            "max_prompts": max_prompts,
            "max_api_failures": max_api_failures,
            "estimated_cost_usd": None,
            "cost_estimation_note": (
                "GPT-4o usage is recorded when the provider returns token metadata. "
                "FAISS embedding calls are estimated separately because they bypass the GPT-4o usage wrapper. "
                "Dollar cost remains null because authoritative current prices are not configured in the repository."
            ),
        },
        "stop_conditions": [
            "repeated API/provider failures",
            "schema/output validation failure",
            "unexpected unapproved mode",
            "secret leakage in output",
            "benchmark harness crash",
        ],
        "stopped_early_reason": stopped_early_reason,
    }
    summary = {
        "run_id": run_id,
        "mode": "live_baseline_ablation",
        "dataset_name": dataset_name,
        "modes": [live_spec.mode.name for live_spec in selected_modes],
        "prompt_categories": category_counts,
        "evaluation_tags": list(BENCHMARK_EVALUATION_TAGS),
        "gold_schema_required": True,
        "metrics": metrics,
        "mode_comparison": mode_comparison,
        "token_usage": token_summary,
        "latency_ms": latency_summary(result.latency_ms for result in results),
        "phase_status": "bounded live baseline/ablation comparison; not a full benchmark",
        "limitations": _baseline_ablation_limitations(),
        "output_files": output_files,
        "stopped_early_reason": stopped_early_reason,
    }
    evidence = {
        "evidence_id": evidence_id,
        "run_id": run_id,
        "date_time_utc": datetime.now(timezone.utc).isoformat(),
        "live_mock_status": "bounded live GPT-4o/LangChain/FAISS proposed-system comparison plus controlled baselines/ablations",
        "model_provider": provider_name,
        "model": model,
        "input_data": {
            "prompts_path": str(prompts_path),
            "reviews_path": str(reviews_path),
            "dataset_name": dataset_name,
            "product_name": product_name,
        },
        "prompt_count": len(prompts),
        "executed_result_count": len(results),
        "prompt_categories": category_counts,
        "mode_comparison": mode_comparison,
        "output_files": output_files,
        "key_results": metrics,
        "token_usage": token_summary,
        "evaluation_tags": list(BENCHMARK_EVALUATION_TAGS),
        "claim_boundary": (
            "Small same-prompt live baseline/ablation comparison for framework components. "
            "Not a full benchmark, not an alternative/open-source model comparison, not a user study, "
            "and not human-annotated ground truth."
        ),
        "limitations": _baseline_ablation_limitations(),
    }
    cost_latency = {
        "run_id": run_id,
        "live_mode": True,
        "model_provider": provider_name,
        "model": model,
        "pre_run_estimated_operations": estimated_calls,
        "token_usage": token_summary,
        "token_usage_by_mode": {
            mode: _token_summary(mode_results)
            for mode, mode_results in _results_by_mode(results).items()
        },
        "estimated_cost_usd": None,
        "cost_estimation_note": "Exact dollar cost unavailable because repository does not configure authoritative current model prices.",
        "latency_ms": latency_summary(result.latency_ms for result in results),
        "latency_ms_by_mode": {
            mode: latency_summary(result.latency_ms for result in mode_results)
            for mode, mode_results in _results_by_mode(results).items()
        },
    }

    manifest_path = run_dir / "manifest.json"
    results_path = run_dir / "results.jsonl"
    summary_path = run_dir / "summary.json"
    evidence_path = run_dir / "evidence.json"
    cost_latency_path = run_dir / "cost_latency.json"
    mode_comparison_json_path = run_dir / "mode_comparison.json"
    mode_comparison_csv_path = run_dir / "mode_comparison.csv"
    failure_examples_path = run_dir / "failure_examples.json"
    output_files.update(
        {
            "manifest": str(manifest_path),
            "results": str(results_path),
            "summary": str(summary_path),
            "evidence": str(evidence_path),
            "cost_latency": str(cost_latency_path),
            "mode_comparison_json": str(mode_comparison_json_path),
            "mode_comparison_csv": str(mode_comparison_csv_path),
            "failure_examples": str(failure_examples_path),
        }
    )
    _write_json(manifest_path, manifest)
    _write_jsonl(results_path, [result.to_dict() for result in results])
    _write_json(summary_path, summary)
    _write_json(evidence_path, evidence)
    _write_json(cost_latency_path, cost_latency)
    _write_json(mode_comparison_json_path, {"run_id": run_id, "rows": mode_comparison})
    _write_mode_comparison_csv(mode_comparison_csv_path, mode_comparison)
    _write_json(failure_examples_path, collect_failure_examples(results_path, max_examples=18))
    return BenchmarkArtifacts(
        run_id=run_id,
        run_dir=run_dir,
        manifest_path=manifest_path,
        results_path=results_path,
        summary_path=summary_path,
        evidence_path=evidence_path,
    )


def estimate_live_operations(prompts: Sequence[BenchmarkPrompt], specs: Sequence[LiveModeSpec]) -> dict[str, Any]:
    by_mode: dict[str, dict[str, int]] = {}
    totals = Counter()
    for spec in specs:
        counts = Counter()
        for prompt in prompts:
            if prompt.expected_route not in spec.mode.supported_routes:
                counts["inapplicable_rows"] += 1
                continue
            if not spec.uses_live_gpt4o:
                counts["deterministic_rows"] += 1
                continue
            if spec.mode.route_strategy == "orchestrator":
                counts["gpt4o_route_calls"] += 1
            elif spec.mode.route_strategy in {"rule_based", "sql_only"}:
                counts["deterministic_route_decisions"] += 1
            if prompt.expected_route == "DIRECT_SQL" and prompt.expected_failure_type != "product_not_found":
                counts["gpt4o_sql_generation_calls"] += 1
            elif prompt.expected_route == "SEMANTICS":
                if spec.uses_faiss:
                    counts["openai_embedding_batch_operations"] += 1
                    counts["gpt4o_semantic_reasoning_calls"] += 1
                else:
                    counts["deterministic_lexical_semantic_rows"] += 1
            elif prompt.expected_route == "ANALYTICS" and prompt.expected_failure_type != "unsupported_chart_type":
                counts["gpt4o_chart_spec_calls"] += 1
        counts["estimated_tracked_gpt4o_calls"] = (
            counts["gpt4o_route_calls"]
            + counts["gpt4o_sql_generation_calls"]
            + counts["gpt4o_semantic_reasoning_calls"]
            + counts["gpt4o_chart_spec_calls"]
        )
        by_mode[spec.mode.name] = dict(sorted(counts.items()))
        totals.update(counts)
    return {
        "by_mode": by_mode,
        "total_estimated_tracked_gpt4o_calls": int(totals["estimated_tracked_gpt4o_calls"]),
        "total_estimated_embedding_batch_operations": int(totals["openai_embedding_batch_operations"]),
        "note": "Estimates are upper-bound planning counts by gold route; provider fallbacks or controlled refusals can reduce actual tracked calls.",
    }


def build_mode_comparison(
    metrics: Mapping[str, Any],
    results: Sequence[BenchmarkResult],
    specs: Sequence[LiveModeSpec],
) -> list[dict[str, Any]]:
    by_mode_metrics = metrics.get("by_mode", {})
    by_mode_results = _results_by_mode(results)
    spec_by_name = {spec.mode.name: spec for spec in specs}
    rows: list[dict[str, Any]] = []
    for mode_name in sorted(by_mode_metrics):
        mode_metrics = by_mode_metrics[mode_name]
        spec = spec_by_name.get(mode_name)
        mode_results = by_mode_results.get(mode_name, [])
        rows.append(
            {
                "mode": mode_name,
                "mode_execution_type": spec.mode_execution_type if spec else None,
                "uses_live_gpt4o": spec.uses_live_gpt4o if spec else None,
                "uses_faiss": spec.uses_faiss if spec else None,
                "uses_sql": spec.uses_sql if spec else None,
                "uses_mock_provider": spec.uses_mock_provider if spec else None,
                "uses_deterministic_logic": spec.uses_deterministic_logic if spec else None,
                "disabled_enrichment_fields": list(spec.disabled_enrichment_fields) if spec else [],
                "prompt_count": mode_metrics.get("prompt_count"),
                "applicable_prompt_count": mode_metrics.get("applicable_prompt_count"),
                "inapplicable_count": mode_metrics.get("inapplicable_count"),
                "overall_success_rate_with_expected_refusals": mode_metrics.get("overall_success_rate_with_expected_refusals"),
                "routing_accuracy": mode_metrics.get("routing_accuracy"),
                "product_table_extraction_accuracy": mode_metrics.get("product_table_extraction_accuracy"),
                "date_range_extraction_accuracy": mode_metrics.get("date_range_extraction_accuracy"),
                "sql_validity_rate_sql_eligible": mode_metrics.get("sql_validity_rate_sql_eligible"),
                "sql_execution_success_rate_sql_executable": mode_metrics.get("sql_execution_success_rate_sql_executable"),
                "evidence_containment_rate_evidence_required": mode_metrics.get("evidence_containment_rate_evidence_required"),
                "answer_fact_proxy_rate_answer_fact_eligible": mode_metrics.get("answer_fact_proxy_rate_answer_fact_eligible"),
                "chart_file_generation_rate_chart_generation_eligible": mode_metrics.get("chart_file_generation_rate_chart_generation_eligible"),
                "chart_numerical_consistency_rate_chart_numeric_eligible": mode_metrics.get("chart_numerical_consistency_rate_chart_numeric_eligible"),
                "expected_failure_handling_rate": mode_metrics.get("expected_refusal_metrics", {})
                .get("overall_expected_failure_handling_rate", {})
                .get("rate"),
                "failure_categories": mode_metrics.get("failure_categories", {}),
                "mean_latency_ms": mode_metrics.get("latency_ms", {}).get("mean"),
                "p50_latency_ms": mode_metrics.get("latency_ms", {}).get("p50"),
                "p95_latency_ms": mode_metrics.get("latency_ms", {}).get("p95"),
                "input_tokens": _sum_token_field(mode_results, "input_tokens"),
                "output_tokens": _sum_token_field(mode_results, "output_tokens"),
                "total_tokens": _sum_token_field(mode_results, "total_tokens"),
                "live_call_count": sum(int(result.live_call_count or 0) for result in mode_results),
                "mock_call_count": sum(int(result.mock_call_count or 0) for result in mode_results),
                "deterministic_call_count": sum(int(result.deterministic_call_count or 0) for result in mode_results),
                "estimated_cost_usd": None,
            }
        )
    return rows


def _with_mode_metadata(result: BenchmarkResult, spec: LiveModeSpec) -> BenchmarkResult:
    return replace(
        result,
        mode_execution_type=spec.mode_execution_type,
        uses_live_gpt4o=spec.uses_live_gpt4o,
        uses_faiss=spec.uses_faiss,
        uses_sql=spec.uses_sql,
        uses_mock_provider=spec.uses_mock_provider,
        uses_deterministic_logic=spec.uses_deterministic_logic,
        disabled_enrichment_fields=spec.disabled_enrichment_fields,
        deterministic_call_count=_deterministic_call_count(result, spec),
        mock_call_count=0,
    )


def _deterministic_call_count(result: BenchmarkResult, spec: LiveModeSpec) -> int:
    count = 0
    if spec.uses_deterministic_logic:
        count += 1
    if result.inapplicable_reason:
        count += 1
    if spec.mode.name == "no_vector_retrieval" and result.expected_route == "SEMANTICS":
        count += 1
    return count


def _mode_description_payload(specs: Sequence[LiveModeSpec]) -> list[dict[str, Any]]:
    return [
        {
            "mode": spec.mode.name,
            "description": spec.mode.description,
            "mode_execution_type": spec.mode_execution_type,
            "uses_live_gpt4o": spec.uses_live_gpt4o,
            "uses_faiss": spec.uses_faiss,
            "uses_sql": spec.uses_sql,
            "uses_mock_provider": spec.uses_mock_provider,
            "uses_deterministic_logic": spec.uses_deterministic_logic,
            "semantic_backend": spec.semantic_backend,
            "disabled_enrichment_fields": list(spec.disabled_enrichment_fields),
            "supported_routes": list(spec.mode.supported_routes),
            "notes": spec.notes,
        }
        for spec in specs
    ]


def _write_mode_comparison_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def _sum_token_field(results: Sequence[BenchmarkResult], field_name: str) -> int | None:
    values = [getattr(result, field_name) for result in results if getattr(result, field_name) is not None]
    if not values:
        return None
    return sum(int(value) for value in values)


def _results_by_mode(results: Sequence[BenchmarkResult]) -> dict[str, list[BenchmarkResult]]:
    grouped: dict[str, list[BenchmarkResult]] = {}
    for result in results:
        grouped.setdefault(result.mode, []).append(result)
    return dict(sorted(grouped.items()))


def _split_mode_names(mode_names: Sequence[str] | None) -> list[str]:
    names: list[str] = []
    for value in mode_names or ():
        names.extend(part.strip() for part in str(value).split(",") if part.strip())
    return names


def _mode_from(base_name: str, *, name: str, description: str | None = None) -> BenchmarkMode:
    base = BENCHMARK_MODES[base_name]
    return replace(base, name=name, description=description or base.description)


LIVE_MODE_SPECS: dict[str, LiveModeSpec] = {
    "proposed_multi_agent_live_gpt4o": LiveModeSpec(
        mode=_mode_from(
            "proposed_multi_agent",
            name="proposed_multi_agent_live_gpt4o",
            description="Paper-aligned live GPT-4o/LangChain/FAISS multi-agent framework.",
        ),
        mode_execution_type="live_gpt4o_langchain_faiss_multi_agent",
        uses_live_gpt4o=True,
        uses_faiss=True,
        uses_sql=True,
        uses_mock_provider=False,
        uses_deterministic_logic=False,
        semantic_backend="faiss",
        notes="Primary proposed-system condition; same framework as the main implementation.",
    ),
    "rule_based_routing_baseline": LiveModeSpec(
        mode=BENCHMARK_MODES["rule_based_routing_baseline"],
        mode_execution_type="hybrid_live_gpt4o_rule_based_routing",
        uses_live_gpt4o=True,
        uses_faiss=True,
        uses_sql=True,
        uses_mock_provider=False,
        uses_deterministic_logic=True,
        semantic_backend="faiss",
        notes="Deterministic routing replaces the orchestrator; downstream SQL/semantic/chart agents remain live where applicable.",
    ),
    "sql_only": LiveModeSpec(
        mode=BENCHMARK_MODES["sql_only"],
        mode_execution_type="hybrid_live_gpt4o_sql_only_baseline",
        uses_live_gpt4o=True,
        uses_faiss=False,
        uses_sql=True,
        uses_mock_provider=False,
        uses_deterministic_logic=True,
        semantic_backend="lexical",
        notes="Only DIRECT_SQL gold items are executable; semantic and analytics prompts are logged as inapplicable.",
    ),
    "sql_plus_semantic_retrieval": LiveModeSpec(
        mode=BENCHMARK_MODES["sql_plus_semantic_retrieval"],
        mode_execution_type="live_gpt4o_sql_plus_faiss_semantic_baseline",
        uses_live_gpt4o=True,
        uses_faiss=True,
        uses_sql=True,
        uses_mock_provider=False,
        uses_deterministic_logic=False,
        semantic_backend="faiss",
        notes="No analytics/chart route; chart prompts are logged as inapplicable.",
    ),
    "no_semantic_tags": LiveModeSpec(
        mode=BENCHMARK_MODES["no_semantic_tags"],
        mode_execution_type="live_gpt4o_faiss_ablation_no_semantic_tags",
        uses_live_gpt4o=True,
        uses_faiss=True,
        uses_sql=True,
        uses_mock_provider=False,
        uses_deterministic_logic=False,
        semantic_backend="faiss",
        disabled_enrichment_fields=BENCHMARK_MODES["no_semantic_tags"].disabled_fields,
        notes="Clears semantic tag fields before loading records.",
    ),
    "no_topic_or_aspect_tags": LiveModeSpec(
        mode=replace(
            BENCHMARK_MODES["no_topic_tags"],
            name="no_topic_or_aspect_tags",
            description="Ablation that clears topic/aspect fields before local retrieval.",
            disabled_fields=("topic", "topic_tags", "aspect", "aspect_tags"),
        ),
        mode_execution_type="live_gpt4o_faiss_ablation_no_topic_or_aspect_tags",
        uses_live_gpt4o=True,
        uses_faiss=True,
        uses_sql=True,
        uses_mock_provider=False,
        uses_deterministic_logic=False,
        semantic_backend="faiss",
        disabled_enrichment_fields=("topic", "topic_tags", "aspect", "aspect_tags"),
        notes="Topic/aspect fields are cleared when present; absent fields are harmless no-ops.",
    ),
    "no_translation": LiveModeSpec(
        mode=BENCHMARK_MODES["no_translation"],
        mode_execution_type="live_gpt4o_faiss_ablation_no_translation",
        uses_live_gpt4o=True,
        uses_faiss=True,
        uses_sql=True,
        uses_mock_provider=False,
        uses_deterministic_logic=False,
        semantic_backend="faiss",
        disabled_enrichment_fields=BENCHMARK_MODES["no_translation"].disabled_fields,
        notes="Clears translated review text and language metadata before loading records.",
    ),
    "no_vector_retrieval": LiveModeSpec(
        mode=BENCHMARK_MODES["no_vector_retrieval"],
        mode_execution_type="hybrid_live_gpt4o_lexical_semantic_no_vector_retrieval",
        uses_live_gpt4o=True,
        uses_faiss=False,
        uses_sql=True,
        uses_mock_provider=False,
        uses_deterministic_logic=True,
        semantic_backend="lexical",
        notes="Disables FAISS/vector retrieval; semantic answers use deterministic lexical retrieval.",
    ),
    "no_orchestrator_simplified_routing": LiveModeSpec(
        mode=BENCHMARK_MODES["no_orchestrator_simplified_routing"],
        mode_execution_type="hybrid_live_gpt4o_simplified_routing_ablation",
        uses_live_gpt4o=True,
        uses_faiss=True,
        uses_sql=True,
        uses_mock_provider=False,
        uses_deterministic_logic=True,
        semantic_backend="faiss",
        notes="Bypasses the live orchestrator router and uses simplified keyword routing.",
    ),
}


def _baseline_ablation_limitations() -> list[str]:
    return [
        "Small 11-prompt programmatically verified benchmark only; not a full benchmark.",
        "Gold items are programmatically verified from local data, not human/adjudicated annotations.",
        "Some baselines and ablations intentionally use deterministic or hybrid logic to isolate framework components.",
        "Single-agent GPT-4o/RAG and alternative/open-source model comparisons remain deferred/future work.",
        "No user study, no inter-annotator agreement, and no human translation-quality evaluation are claimed.",
        "Dollar cost is not computed without authoritative current pricing configured in the repository.",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a bounded live baseline/ablation comparison over programmatic gold prompts.")
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS_PATH)
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dataset-name", default="amazon_all_beauty_programmatic_gold_live_baseline_ablation")
    parser.add_argument("--product-name", default="amazon all beauty")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER, choices=("langchain", "openai"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-prompts", type=int, default=11)
    parser.add_argument("--max-api-failures", type=int, default=3)
    parser.add_argument("--evidence-id", default=DEFAULT_EVIDENCE_ID)
    parser.add_argument("--modes", nargs="*", default=None, help="Mode names, comma-separated names, or all.")
    args = parser.parse_args()

    artifacts = run_live_baseline_ablation(
        prompts_path=args.prompts,
        reviews_path=args.reviews,
        output_dir=args.output_dir,
        run_id=args.run_id,
        dataset_name=args.dataset_name,
        product_name=args.product_name,
        provider_name=args.provider,
        model=args.model,
        max_prompts=args.max_prompts,
        max_api_failures=args.max_api_failures,
        evidence_id=args.evidence_id,
        mode_names=args.modes,
    )
    print(f"run_id={artifacts.run_id}")
    print(f"manifest={artifacts.manifest_path}")
    print(f"results={artifacts.results_path}")
    print(f"summary={artifacts.summary_path}")
    print(f"evidence={artifacts.evidence_path}")


if __name__ == "__main__":
    main()

