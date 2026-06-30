"""Legacy/auxiliary live model comparison runner.

This script is retained for backward compatibility with earlier bounded
model-substitution artifacts. The manuscript-aligned cross-model workflow
evaluation is implemented in ``evaluation/model_interface_robustness.py`` and
uses the fixed 30-prompt configuration reported in the paper.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import replace
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
from evaluation.benchmark_schema import BenchmarkArtifacts, BenchmarkResult
from evaluation.failure_examples import collect_failure_examples
from evaluation.live_pilot import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROMPTS_PATH,
    DEFAULT_REVIEWS_PATH,
    UsageTrackingProvider,
    _git_commit_sha,
    _install_process_environment,
    _prompt_had_api_failure,
    _scrubbed_provider_config,
    _token_summary,
    read_dotenv,
)
from evaluation.model_config import ModelSubstitutionConfig, load_model_configs, validate_api_keys, validate_endpoint_urls
from evaluation.run_benchmark import (
    BENCHMARK_MODES,
    BENCHMARK_EVALUATION_TAGS,
    _run_prompt,
    _write_json,
    _write_jsonl,
    load_gold_benchmark_prompts,
)
from llm_review_analysis.agents import RetrievalAgent, ReviewOrchestrator
from llm_review_analysis.config import ensure_directories, load_settings
from llm_review_analysis.providers import build_llm_provider


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "evaluation" / "model_configs.json"
DEFAULT_EVIDENCE_ID = "EVID-LIVE-MODEL-SUBSTITUTION-001"
DEFAULT_RUN_ID_PREFIX = "live_model_substitution"
TRACKED_PROVIDER_CONFIG_KEYS = (
    "ALLOW_LIVE_LLM",
    "ALLOW_LIVE_RETRIEVAL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_MODEL_ID",
    "EMBEDDING_MODEL",
    "HF_ENDPOINT_URL",
    "HF_LLAMA_ENDPOINT_URL",
    "HF_QWEN_ENDPOINT_URL",
    "HF_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
    "LLM_MAX_RETRIES",
    "LLM_MAX_TOKENS",
    "LLM_MODEL",
    "LLM_PROVIDER",
    "LLM_TEMPERATURE",
    "LLM_TIMEOUT_SECONDS",
    "OPENAI_API_KEY",
    "SEMANTIC_RETRIEVAL_BACKEND",
)


def run_live_model_substitution(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    prompts_path: str | Path = DEFAULT_PROMPTS_PATH,
    reviews_path: str | Path = DEFAULT_REVIEWS_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    run_id: str | None = None,
    dataset_name: str = "amazon_all_beauty_programmatic_gold_live_model_substitution",
    product_name: str = "amazon all beauty",
    model_labels: Sequence[str] | None = None,
    semantic_backend: str = "faiss",
    max_prompts: int = 11,
    max_api_failures: int = 3,
    evidence_id: str = DEFAULT_EVIDENCE_ID,
) -> BenchmarkArtifacts:
    config_path = Path(config_path)
    prompts_path = Path(prompts_path)
    reviews_path = Path(reviews_path)
    output_root = Path(output_dir)
    run_id = run_id or f"{DEFAULT_RUN_ID_PREFIX}_{datetime.now(timezone.utc).strftime('%Y%m%d')}_{uuid4().hex[:4]}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    dotenv_values = read_dotenv(PROJECT_ROOT / ".env")
    runtime_env = dict(os.environ)
    runtime_env.update(dotenv_values)
    configs = load_model_configs(config_path, selected_labels=model_labels, env=runtime_env, require_model_ids=True)
    missing_keys = _missing_runtime_keys(configs, runtime_env, semantic_backend=semantic_backend)
    if missing_keys:
        raise RuntimeError("Missing required API key(s): " + json.dumps(missing_keys, sort_keys=True))

    prompts = [
        prompt
        for prompt in load_gold_benchmark_prompts(prompts_path)
        if prompt.gold_verification_status == "programmatically_verified"
    ][:max_prompts]
    review_rows = json.loads(reviews_path.read_text(encoding="utf-8"))
    mode_spec = BENCHMARK_MODES["proposed_multi_agent"]
    results: list[BenchmarkResult] = []
    api_failure_count = 0
    stopped_early_reason: str | None = None

    for config in configs:
        model_mode_name = _model_mode_name(config)
        model_env = dict(dotenv_values)
        model_env.update(
            {
                "LLM_REVIEW_PROJECT_ROOT": str(run_dir),
                "REVIEWS_DB_PATH": str(run_dir / "runtime" / f"{model_mode_name}.db"),
                "OUTPUT_DIR": str(run_dir / "charts" / model_mode_name),
                "VECTORSTORE_DIR": str(run_dir / "vectorstores" / model_mode_name),
                "LLM_PROVIDER": config.provider,
                "LLM_MODEL": config.model_id,
                "LLM_TEMPERATURE": str(config.temperature),
                "LLM_MAX_TOKENS": str(config.max_tokens),
                "LLM_TIMEOUT_SECONDS": str(config.timeout_seconds),
                "LLM_MAX_RETRIES": str(config.max_retries),
                "SEMANTIC_RETRIEVAL_BACKEND": semantic_backend,
                "ALLOW_LIVE_LLM": "true",
                "ALLOW_LIVE_RETRIEVAL": "false",
            }
        )
        if config.endpoint_url_env and runtime_env.get(config.endpoint_url_env):
            model_env["HF_ENDPOINT_URL"] = runtime_env[config.endpoint_url_env]
        _install_process_environment(model_env)
        settings = load_settings(model_env)
        ensure_directories(settings)
        wrapped_provider = build_llm_provider(settings)
        provider = UsageTrackingProvider(
            wrapped_provider,
            provider_name=config.provider,
            model=config.model_id,
        )
        with sqlite3.connect(settings.database_path) as conn:
            conn.row_factory = sqlite3.Row
            RetrievalAgent(settings).load_records(conn, product_name, review_rows)
            orchestrator = ReviewOrchestrator(settings, provider)
            for prompt in prompts:
                provider.start_prompt(f"{config.model_label}:{prompt.prompt_id}")
                result = _run_prompt(run_id, mode_spec, orchestrator, conn, prompt, provider.model)
                usage = provider.finish_prompt()
                result = _with_model_substitution_usage(
                    result,
                    config=config,
                    usage=usage,
                    semantic_backend=semantic_backend,
                )
                results.append(result)
                if _prompt_had_api_failure(usage):
                    api_failure_count += 1
                if api_failure_count >= max_api_failures:
                    stopped_early_reason = f"Stopped after {api_failure_count} provider/API failure(s)."
                    break
        if stopped_early_reason:
            break

    metrics = summarize_benchmark_results(results)
    model_comparison = build_model_comparison(results, configs)
    token_summary = _token_summary(results)
    category_counts = dict(sorted(Counter(result.category for result in results).items()))
    output_files: dict[str, str] = {}
    manifest = {
        "run_id": run_id,
        "mode": "live_model_substitution",
        "dataset_name": dataset_name,
        "prompts_path": str(prompts_path),
        "reviews_path": str(reviews_path),
        "config_path": str(config_path),
        "output_dir": str(run_dir),
        "prompt_count": len(prompts),
        "executed_result_count": len(results),
        "live_mode": True,
        "model_configs": [_model_config_manifest(config, runtime_env) for config in configs],
        "provider_config": _visible_provider_config(runtime_env),
        "semantic_retrieval_backend": semantic_backend,
        "command": " ".join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "code_version": _git_commit_sha(),
        "evaluation_tags": list(BENCHMARK_EVALUATION_TAGS) + ["model-substitution"],
        "gold_schema_required": True,
        "gold_inclusion_rule": "programmatically_verified prompts only; author/human-unverified judgment prompts excluded",
        "fairness_rules": _fairness_rules(),
        "stopped_early_reason": stopped_early_reason,
    }
    summary = {
        "run_id": run_id,
        "mode": "live_model_substitution",
        "dataset_name": dataset_name,
        "prompt_categories": category_counts,
        "evaluation_tags": list(BENCHMARK_EVALUATION_TAGS) + ["model-substitution"],
        "gold_schema_required": True,
        "metrics": metrics,
        "model_comparison": model_comparison,
        "token_usage": token_summary,
        "latency_ms": latency_summary(result.latency_ms for result in results),
        "phase_status": "prepared bounded live cross-model workflow run; not a model leaderboard",
        "limitations": _model_substitution_limitations(),
        "output_files": output_files,
        "stopped_early_reason": stopped_early_reason,
    }
    evidence = {
        "evidence_id": evidence_id,
        "run_id": run_id,
        "date_time_utc": datetime.now(timezone.utc).isoformat(),
        "live_mock_status": "live cross-model workflow run",
        "input_data": {
            "prompts_path": str(prompts_path),
            "reviews_path": str(reviews_path),
            "dataset_name": dataset_name,
            "product_name": product_name,
        },
        "prompt_count": len(prompts),
        "executed_result_count": len(results),
        "model_comparison": model_comparison,
        "output_files": output_files,
        "key_results": metrics,
        "token_usage": token_summary,
        "evaluation_tags": list(BENCHMARK_EVALUATION_TAGS) + ["model-substitution"],
        "claim_boundary": (
            "Legacy cross-model workflow evidence only. GPT-4o remains the primary implementation model; "
            "additional models test portability/sensitivity of the same workflow, not broad model superiority."
        ),
        "limitations": _model_substitution_limitations(),
    }
    cost_latency = {
        "run_id": run_id,
        "mode": "live_model_substitution",
        "token_usage": token_summary,
        "estimated_cost_usd": None,
        "cost_estimation_note": "Dollar cost remains null unless authoritative provider pricing is configured locally.",
        "latency_ms": latency_summary(result.latency_ms for result in results),
        "by_model": {
            row["model_label"]: {
                "provider": row["provider"],
                "model_id": row["model_id"],
                "token_usage": row["token_usage"],
                "latency_ms": row["latency_ms"],
            }
            for row in model_comparison
        },
    }

    manifest_path = run_dir / "manifest.json"
    results_path = run_dir / "results.jsonl"
    summary_path = run_dir / "summary.json"
    model_comparison_json_path = run_dir / "model_comparison.json"
    model_comparison_csv_path = run_dir / "model_comparison.csv"
    cost_latency_path = run_dir / "cost_latency.json"
    failure_examples_path = run_dir / "failure_examples.json"
    evidence_path = run_dir / "evidence.json"
    output_files.update(
        {
            "manifest": str(manifest_path),
            "results": str(results_path),
            "summary": str(summary_path),
            "model_comparison_json": str(model_comparison_json_path),
            "model_comparison_csv": str(model_comparison_csv_path),
            "cost_latency": str(cost_latency_path),
            "failure_examples": str(failure_examples_path),
            "evidence": str(evidence_path),
        }
    )
    _write_json(manifest_path, manifest)
    _write_jsonl(results_path, [result.to_dict() for result in results])
    _write_json(summary_path, summary)
    _write_json(model_comparison_json_path, {"run_id": run_id, "rows": model_comparison})
    _write_model_comparison_csv(model_comparison_csv_path, model_comparison)
    _write_json(cost_latency_path, cost_latency)
    _write_json(failure_examples_path, collect_failure_examples(results_path, max_examples=18))
    _write_json(evidence_path, evidence)
    return BenchmarkArtifacts(
        run_id=run_id,
        run_dir=run_dir,
        manifest_path=manifest_path,
        results_path=results_path,
        summary_path=summary_path,
        evidence_path=evidence_path,
    )


def build_model_comparison(
    results: Sequence[BenchmarkResult],
    configs: Sequence[ModelSubstitutionConfig],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config in configs:
        mode_name = _model_mode_name(config)
        model_results = [result for result in results if result.mode == mode_name]
        metrics = summarize_benchmark_results(model_results)
        rows.append(
            {
                "model_label": config.model_label,
                "provider": config.provider,
                "model_id": config.model_id,
                "report_model_id": config.report_model_id or config.model_id,
                "endpoint_url_env": config.endpoint_url_env,
                "endpoint_url_status": _endpoint_url_status(config, os.environ),
                "role": config.role,
                "model_family": config.model_family,
                "prompt_count": len(model_results),
                "overall_success_rate_with_expected_refusals": metrics.get("overall_success_rate_with_expected_refusals"),
                "routing_accuracy": metrics.get("routing_accuracy"),
                "sql_validity_rate_sql_eligible": metrics.get("sql_validity_rate_sql_eligible"),
                "sql_execution_success_rate_sql_executable": metrics.get("sql_execution_success_rate_sql_executable"),
                "answer_fact_proxy_rate_answer_fact_eligible": metrics.get("answer_fact_proxy_rate_answer_fact_eligible"),
                "evidence_containment_rate_evidence_required": metrics.get("evidence_containment_rate_evidence_required"),
                "chart_type_accuracy": metrics.get("chart_type_accuracy"),
                "chart_numerical_consistency_rate_chart_numeric_eligible": metrics.get("chart_numerical_consistency_rate_chart_numeric_eligible"),
                "expected_failure_handling_rate": metrics.get("expected_failure_handling_rate"),
                "failure_categories": metrics.get("failure_categories", {}),
                "latency_ms": metrics.get("latency_ms", {}),
                "token_usage": _token_summary(list(model_results)),
            }
        )
    return rows


def dry_config_check(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    model_labels: Sequence[str] | None = None,
    semantic_backend: str = "faiss",
) -> dict[str, Any]:
    dotenv_values = read_dotenv(PROJECT_ROOT / ".env")
    runtime_env = dict(os.environ)
    runtime_env.update(dotenv_values)
    configs = load_model_configs(config_path, selected_labels=model_labels, env=runtime_env, require_model_ids=False)
    return {
        "config_path": str(config_path),
        "semantic_backend": semantic_backend,
        "models": [_model_config_manifest(config, runtime_env) for config in configs],
        "missing_runtime_keys": _missing_runtime_keys(configs, runtime_env, semantic_backend=semantic_backend),
        "provider_config": _visible_provider_config(runtime_env),
        "live_calls_made": False,
    }


def preflight_check(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    prompts_path: str | Path = DEFAULT_PROMPTS_PATH,
    reviews_path: str | Path = DEFAULT_REVIEWS_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    model_labels: Sequence[str] | None = None,
    semantic_backend: str = "faiss",
    expected_prompt_count: int = 11,
) -> dict[str, Any]:
    dotenv_path = PROJECT_ROOT / ".env"
    dotenv_values = read_dotenv(dotenv_path)
    runtime_env = dict(os.environ)
    runtime_env.update(dotenv_values)
    configs = load_model_configs(config_path, selected_labels=model_labels, env=runtime_env, require_model_ids=False)
    prompts = [
        prompt
        for prompt in load_gold_benchmark_prompts(prompts_path)
        if prompt.gold_verification_status == "programmatically_verified"
    ] if Path(prompts_path).exists() else []
    reviews_valid, review_count = _review_fixture_status(reviews_path)
    output_createable = _can_create_output_dir(output_dir)
    missing_runtime_keys = _missing_runtime_keys(configs, runtime_env, semantic_backend=semantic_backend)
    required_models = _required_model_label_status(configs)
    safe_defaults = _safe_default_status(runtime_env)
    missing_variables = _missing_preflight_variables(
        runtime_env,
        missing_runtime_keys=missing_runtime_keys,
        required_models=required_models,
        safe_defaults=safe_defaults,
        prompts_ok=len(prompts) == expected_prompt_count,
        reviews_valid=reviews_valid,
        output_createable=output_createable,
    )
    ready = not missing_variables
    return {
        "phase": "preflight",
        "live_calls_made": False,
        "benchmark_prompts_sent": False,
        "dotenv_exists": dotenv_path.exists(),
        "safe_defaults": safe_defaults,
        "required_key_status": {
            "OPENAI_API_KEY": _redacted_presence(runtime_env, "OPENAI_API_KEY"),
            "ANTHROPIC_API_KEY": _redacted_presence(runtime_env, "ANTHROPIC_API_KEY"),
            "HF_TOKEN_OR_HUGGINGFACEHUB_API_TOKEN": (
                "<present redacted>"
                if runtime_env.get("HF_TOKEN") or runtime_env.get("HUGGINGFACEHUB_API_TOKEN")
                else "<missing>"
            ),
        },
        "endpoint_url_status": {
            "HF_LLAMA_ENDPOINT_URL": _redacted_presence(runtime_env, "HF_LLAMA_ENDPOINT_URL"),
            "HF_QWEN_ENDPOINT_URL": _redacted_presence(runtime_env, "HF_QWEN_ENDPOINT_URL"),
        },
        "model_label_status": required_models,
        "model_configs": [_model_config_manifest(config, runtime_env) for config in configs],
        "benchmark_prompt_count": len(prompts),
        "expected_prompt_count": expected_prompt_count,
        "prompts_path": str(prompts_path),
        "reviews_path": str(reviews_path),
        "review_fixture_valid": reviews_valid,
        "review_fixture_count": review_count,
        "configured_reviews_db_path": runtime_env.get("REVIEWS_DB_PATH", ""),
        "output_dir": str(output_dir),
        "output_folder_createable": output_createable,
        "secrets_redacted": True,
        "missing_runtime_keys": missing_runtime_keys,
        "missing_variables_or_conditions": missing_variables,
        "ready_for_manual_endpoint_activation": ready,
        "status_message": (
            "Preflight passed. Ready for manual endpoint activation."
            if ready
            else "Preflight not ready. Resolve missing variables or conditions before endpoint activation."
        ),
    }


def _with_model_substitution_usage(
    result: BenchmarkResult,
    *,
    config: ModelSubstitutionConfig,
    usage: dict[str, Any],
    semantic_backend: str,
) -> BenchmarkResult:
    return replace(
        result,
        mode=_model_mode_name(config),
        mode_name=_model_mode_name(config),
        model_provider=config.provider,
        model=config.model_id,
        input_tokens=_optional_int(usage.get("input_tokens")),
        output_tokens=_optional_int(usage.get("output_tokens")),
        total_tokens=_optional_int(usage.get("total_tokens")),
        estimated_cost_usd=None,
        token_usage=usage,
        mode_execution_type=f"live_model_substitution:{config.provider}",
        uses_live_gpt4o=config.model_id == "gpt-4o",
        uses_faiss=semantic_backend == "faiss",
        uses_mock_provider=False,
        live_call_count=_optional_int(usage.get("call_count")) or 0,
        mock_call_count=0,
    )


def _missing_runtime_keys(
    configs: Sequence[ModelSubstitutionConfig],
    env: Mapping[str, str],
    *,
    semantic_backend: str,
) -> dict[str, list[str]]:
    missing = validate_api_keys(configs, env)
    for model_label, keys in validate_endpoint_urls(configs, env).items():
        missing.setdefault(model_label, []).extend(keys)
    if semantic_backend == "faiss" and not env.get("OPENAI_API_KEY"):
        missing.setdefault("faiss_embeddings", []).append("OPENAI_API_KEY")
    return missing


def _model_mode_name(config: ModelSubstitutionConfig) -> str:
    return f"proposed_multi_agent__{config.model_label}"


def _model_config_manifest(config: ModelSubstitutionConfig, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    payload = config.to_dict()
    payload["api_key_status"] = "<present redacted if configured at runtime>"
    payload["endpoint_url_status"] = _endpoint_url_status(config, env or {})
    return payload


def _endpoint_url_status(config: ModelSubstitutionConfig, env: Mapping[str, str]) -> str | None:
    if not config.endpoint_url_env:
        return None
    return "<present redacted>" if env.get(config.endpoint_url_env) else "<missing>"


def _visible_provider_config(values: Mapping[str, str]) -> dict[str, Any]:
    return _scrubbed_provider_config(
        {
            key: str(values[key])
            for key in TRACKED_PROVIDER_CONFIG_KEYS
            if key in values and values[key] is not None
        }
    )


def _redacted_presence(values: Mapping[str, str], key: str) -> str:
    return "<present redacted>" if values.get(key) else "<missing>"


def _review_fixture_status(path: str | Path) -> tuple[bool, int | None]:
    fixture_path = Path(path)
    if not fixture_path.exists():
        return False, None
    try:
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    except Exception:
        return False, None
    return isinstance(payload, list), len(payload) if isinstance(payload, list) else None


def _can_create_output_dir(path: str | Path) -> bool:
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
    except Exception:
        return False
    return True


def _required_model_label_status(configs: Sequence[ModelSubstitutionConfig]) -> dict[str, bool]:
    labels = {config.model_label for config in configs}
    return {
        "GPT-4o": "gpt4o_primary" in labels,
        "Claude Sonnet": "claude_sonnet_4_6_configured" in labels,
        "Llama-3.3-70B-Instruct GGUF endpoint": "llama3_3_70b_instruct_endpoint" in labels,
        "Qwen2.5-72B-Instruct endpoint": "qwen2_5_72b_instruct_endpoint" in labels,
    }


def _safe_default_status(values: Mapping[str, str]) -> dict[str, bool]:
    return {
        "LLM_PROVIDER=mock": values.get("LLM_PROVIDER") == "mock",
        "ALLOW_LIVE_LLM=false": values.get("ALLOW_LIVE_LLM") == "false",
        "ALLOW_LIVE_RETRIEVAL=false": values.get("ALLOW_LIVE_RETRIEVAL") == "false",
        "SEMANTIC_RETRIEVAL_BACKEND=lexical": values.get("SEMANTIC_RETRIEVAL_BACKEND") == "lexical",
    }


def _missing_preflight_variables(
    values: Mapping[str, str],
    *,
    missing_runtime_keys: Mapping[str, list[str]],
    required_models: Mapping[str, bool],
    safe_defaults: Mapping[str, bool],
    prompts_ok: bool,
    reviews_valid: bool,
    output_createable: bool,
) -> list[str]:
    missing: list[str] = []
    if not (PROJECT_ROOT / ".env").exists():
        missing.append(".env")
    if not values.get("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if not values.get("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY")
    if not (values.get("HF_TOKEN") or values.get("HUGGINGFACEHUB_API_TOKEN")):
        missing.append("HF_TOKEN or HUGGINGFACEHUB_API_TOKEN")
    for key in ("HF_LLAMA_ENDPOINT_URL", "HF_QWEN_ENDPOINT_URL"):
        if not values.get(key):
            missing.append(key)
    for name, ok in safe_defaults.items():
        if not ok:
            missing.append(name)
    for name, ok in required_models.items():
        if not ok:
            missing.append(f"model config: {name}")
    if missing_runtime_keys:
        for keys in missing_runtime_keys.values():
            for key in keys:
                if key not in missing:
                    missing.append(key)
    if not prompts_ok:
        missing.append("benchmark prompt count must be 11")
    if not reviews_valid:
        missing.append("review fixture/database path must be readable")
    if not output_createable:
        missing.append("output folder must be createable")
    return sorted(set(missing))


def _optional_int(value: Any) -> int | None:
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _write_model_comparison_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def _fairness_rules() -> list[str]:
    return [
        "Same workflow architecture for all models.",
        "Same prompts and gold answers.",
        "Same database and retrieved records.",
        "Same eligibility-aware metrics.",
        "Same expected-refusal scoring.",
        "Same temperature where possible.",
        "Same maximum output budget where possible.",
        "Same timeout/retry policy where possible.",
        "Model-specific prompt formatting allowed only when required by API format.",
        "No manual correction of outputs.",
        "No changing gold answers after seeing model outputs.",
        "Report API failures separately from task failures.",
        "Report token/cost availability differences transparently.",
    ]


def _model_substitution_limitations() -> list[str]:
    return [
        "Legacy cross-model workflow analysis only; not a comprehensive model leaderboard.",
        "GPT-4o remains the primary implementation model for the workflow.",
        "Run uses the same bounded programmatically verified prompt set unless a larger verified-gold set is approved.",
        "Token usage is recorded only when providers return usage metadata; otherwise fields remain null.",
        "Dollar cost remains null unless authoritative provider pricing is configured locally.",
        "No claim of broad model-agnostic superiority or independence from model capability.",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run or dry-check the legacy bounded live model comparison script. "
            "Use evaluation/model_interface_robustness.py for the manuscript-aligned 30-prompt cross-model workflow evaluation."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS_PATH)
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dataset-name", default="amazon_all_beauty_programmatic_gold_live_model_substitution")
    parser.add_argument("--product-name", default="amazon all beauty")
    parser.add_argument("--model-labels", nargs="*", default=None)
    parser.add_argument("--semantic-backend", default="faiss", choices=("faiss", "lexical"))
    parser.add_argument("--max-prompts", type=int, default=11)
    parser.add_argument("--max-api-failures", type=int, default=3)
    parser.add_argument("--evidence-id", default=DEFAULT_EVIDENCE_ID)
    parser.add_argument("--dry-config-check", action="store_true", help="Validate model config/key presence without sending benchmark prompts.")
    parser.add_argument("--preflight-check", action="store_true", help="Run the no-call legacy preflight readiness check.")
    args = parser.parse_args()

    if args.preflight_check:
        print(
            json.dumps(
                preflight_check(
                    config_path=args.config,
                    prompts_path=args.prompts,
                    reviews_path=args.reviews,
                    output_dir=args.output_dir,
                    model_labels=args.model_labels,
                    semantic_backend=args.semantic_backend,
                    expected_prompt_count=args.max_prompts,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.dry_config_check:
        print(json.dumps(dry_config_check(config_path=args.config, model_labels=args.model_labels, semantic_backend=args.semantic_backend), indent=2, sort_keys=True))
        return

    artifacts = run_live_model_substitution(
        config_path=args.config,
        prompts_path=args.prompts,
        reviews_path=args.reviews,
        output_dir=args.output_dir,
        run_id=args.run_id,
        dataset_name=args.dataset_name,
        product_name=args.product_name,
        model_labels=args.model_labels,
        semantic_backend=args.semantic_backend,
        max_prompts=args.max_prompts,
        max_api_failures=args.max_api_failures,
        evidence_id=args.evidence_id,
    )
    print(f"run_id={artifacts.run_id}")
    print(f"manifest={artifacts.manifest_path}")
    print(f"results={artifacts.results_path}")
    print(f"summary={artifacts.summary_path}")
    print(f"evidence={artifacts.evidence_path}")


if __name__ == "__main__":
    main()

