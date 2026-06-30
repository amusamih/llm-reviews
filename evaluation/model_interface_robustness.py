from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import time
from typing import Any, Mapping, Sequence
from urllib import error as urllib_error
from urllib import request as urllib_request
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (SRC_ROOT, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from evaluation.live_pilot import (  # noqa: E402
    UsageTrackingProvider,
    _git_commit_sha,
    _install_process_environment,
    _scrubbed_provider_config,
    read_dotenv,
)
from evaluation.model_config import ModelSubstitutionConfig, load_model_configs  # noqa: E402
from evaluation.run_benchmark import BENCHMARK_MODES, _execute_route  # noqa: E402
from llm_review_analysis.agents import RetrievalAgent, ReviewOrchestrator  # noqa: E402
from llm_review_analysis.analytics.chart_specs import CHART_TYPES  # noqa: E402
from llm_review_analysis.config import Settings, ensure_directories, load_settings  # noqa: E402
from llm_review_analysis.db.schema import REVIEW_COLUMNS  # noqa: E402
from llm_review_analysis.db.sql_validator import SQLValidationError, validate_select_sql  # noqa: E402
from llm_review_analysis.providers import build_llm_provider  # noqa: E402


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "evaluation" / "model_configs.json"
DEFAULT_PROMPTS_PATH = PROJECT_ROOT / "evaluation" / "interface_robustness_prompts.json"
DEFAULT_REVIEWS_PATH = PROJECT_ROOT / "outputs" / "programmatic_gold" / "amazon_all_beauty_20260624" / "programmatic_gold_reviews.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "benchmarks"
DEFAULT_PRODUCT_NAME = "amazon all beauty"
DEFAULT_DATASET_NAME = "amazon_all_beauty_250_programmatic_artifact"
DEFAULT_PREFLIGHT_MODELS = ("gpt4o", "claude", "llama", "qwen")
DEFAULT_AGENT_SCOPES = ("full",)
MODEL_ALIASES = {
    "gpt4o": "gpt4o_primary",
    "gpt-4o": "gpt4o_primary",
    "gpt4o_primary": "gpt4o_primary",
    "claude": "claude_sonnet_4_6_configured",
    "sonnet": "claude_sonnet_4_6_configured",
    "claude_sonnet_4_6_configured": "claude_sonnet_4_6_configured",
    "qwen": "qwen2_5_72b_instruct_endpoint",
    "qwen2.5": "qwen2_5_72b_instruct_endpoint",
    "qwen2_5_72b_instruct_endpoint": "qwen2_5_72b_instruct_endpoint",
    "llama": "llama3_3_70b_instruct_endpoint",
    "llama3.3": "llama3_3_70b_instruct_endpoint",
    "llama3_3_70b_instruct_endpoint": "llama3_3_70b_instruct_endpoint",
}
MODEL_DISPLAY_NAMES = {
    "gpt4o_primary": "GPT-4o",
    "claude_sonnet_4_6_configured": "Claude Sonnet",
    "llama3_3_70b_instruct_endpoint": "Llama-3.3-70B-Instruct",
    "qwen2_5_72b_instruct_endpoint": "Qwen2.5-72B-Instruct",
}
LOCAL_MODEL_LABELS = {"llama3_3_70b_instruct_endpoint", "qwen2_5_72b_instruct_endpoint"}
AGENT_SCOPES = {"full", "orchestrator", "semantics", "analytics"}
EXPECTED_ROUTES = {"DIRECT_SQL", "SEMANTICS", "ANALYTICS", "REFUSAL"}
FAILURE_TYPES = {
    "none",
    "routing_error",
    "reasoning_error",
    "invalid_sql",
    "invalid_chart_spec",
    "refusal_error",
    "unsupported_overanswer",
    "missing_evidence",
    "api_error",
    "timeout",
    "parse_error",
}
TRACKED_PROVIDER_CONFIG_KEYS = (
    "ALLOW_LIVE_LLM",
    "ALLOW_LIVE_RETRIEVAL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_MODEL_ID",
    "EMBEDDING_MODEL",
    "HF_ENDPOINT_URL",
    "HF_LLAMA_ENDPOINT_URL",
    "HF_MODEL_LLAMA",
    "HF_MODEL_LLAMA_REPORT_LABEL",
    "HF_QWEN_ENDPOINT_URL",
    "HF_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
    "LLM_MAX_RETRIES",
    "LLM_MAX_TOKENS",
    "LLM_MODEL",
    "LLM_PROVIDER",
    "LLM_TEMPERATURE",
    "LLM_TIMEOUT_SECONDS",
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "SEMANTIC_RETRIEVAL_BACKEND",
)
CLOUD_PROVIDER_NAMES = {"langchain", "langchain-openai", "langchain_openai", "openai", "anthropic", "claude"}
PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "http_proxy",
    "https_proxy",
    "ALL_PROXY",
    "all_proxy",
    "NO_PROXY",
    "no_proxy",
)
BASE_URL_ENV_KEYS = ("OPENAI_BASE_URL", "ANTHROPIC_BASE_URL")
PROJECT_ENDPOINT_ENV_KEYS = ("HF_ENDPOINT_URL", "HF_LLAMA_ENDPOINT_URL", "HF_QWEN_ENDPOINT_URL")


@dataclass(frozen=True)
class InterfacePrompt:
    prompt_id: str
    prompt_text: str
    capability_category: str
    expected_route: str
    expected_behavior: str
    expected_refusal: bool
    validation_rule: str
    notes: str = ""
    language: str = "en"
    product: str | None = None
    expected_product_table: str | None = None
    expected_date_range: str | None = None
    expected_result_type: str | None = None
    expected_answer_contains: tuple[str, ...] = ()
    expected_answer_any: tuple[str, ...] = ()
    expected_evidence_contains: tuple[str, ...] = ()
    expected_evidence_any: tuple[str, ...] = ()
    expected_sql_pattern: str | None = None
    expected_chart_type: str | None = None
    expected_chart_grouping: str | None = None
    expected_chart_values: dict[str, float] = field(default_factory=dict)
    expected_failure_type: str | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "InterfacePrompt":
        missing = [
            key
            for key in (
                "prompt_id",
                "prompt_text",
                "capability_category",
                "expected_route",
                "expected_behavior",
                "expected_refusal",
                "validation_rule",
                "notes",
            )
            if key not in raw
        ]
        if missing:
            raise ValueError(f"Prompt {raw.get('prompt_id', '<unknown>')} is missing required field(s): {', '.join(missing)}")
        expected_route = str(raw["expected_route"]).strip().upper()
        if expected_route not in EXPECTED_ROUTES:
            raise ValueError(f"Unsupported expected_route for {raw.get('prompt_id')}: {expected_route}")
        chart_values = raw.get("expected_chart_values") or {}
        if not isinstance(chart_values, Mapping):
            raise ValueError(f"expected_chart_values must be an object for {raw.get('prompt_id')}")
        return cls(
            prompt_id=str(raw["prompt_id"]),
            prompt_text=str(raw["prompt_text"]),
            capability_category=str(raw["capability_category"]),
            expected_route=expected_route,
            expected_behavior=str(raw["expected_behavior"]),
            expected_refusal=bool(raw["expected_refusal"]),
            validation_rule=str(raw["validation_rule"]),
            notes=str(raw.get("notes", "")),
            language=str(raw.get("language", "en")),
            product=_optional_str(raw.get("product")),
            expected_product_table=_optional_str(raw.get("expected_product_table")),
            expected_date_range=_optional_str(raw.get("expected_date_range")),
            expected_result_type=_optional_str(raw.get("expected_result_type")),
            expected_answer_contains=tuple(str(value) for value in raw.get("expected_answer_contains", ())),
            expected_answer_any=tuple(str(value) for value in raw.get("expected_answer_any", ())),
            expected_evidence_contains=tuple(str(value) for value in raw.get("expected_evidence_contains", ())),
            expected_evidence_any=tuple(str(value) for value in raw.get("expected_evidence_any", ())),
            expected_sql_pattern=_optional_str(raw.get("expected_sql_pattern")),
            expected_chart_type=_optional_str(raw.get("expected_chart_type")),
            expected_chart_grouping=_optional_str(raw.get("expected_chart_grouping")),
            expected_chart_values={str(key): float(value) for key, value in chart_values.items()},
            expected_failure_type=_optional_str(raw.get("expected_failure_type")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "prompt_text": self.prompt_text,
            "capability_category": self.capability_category,
            "expected_route": self.expected_route,
            "expected_behavior": self.expected_behavior,
            "expected_refusal": self.expected_refusal,
            "validation_rule": self.validation_rule,
            "notes": self.notes,
            "language": self.language,
            "product": self.product,
            "expected_product_table": self.expected_product_table,
            "expected_date_range": self.expected_date_range,
            "expected_result_type": self.expected_result_type,
            "expected_answer_contains": list(self.expected_answer_contains),
            "expected_answer_any": list(self.expected_answer_any),
            "expected_evidence_contains": list(self.expected_evidence_contains),
            "expected_evidence_any": list(self.expected_evidence_any),
            "expected_sql_pattern": self.expected_sql_pattern,
            "expected_chart_type": self.expected_chart_type,
            "expected_chart_grouping": self.expected_chart_grouping,
            "expected_chart_values": dict(self.expected_chart_values),
            "expected_failure_type": self.expected_failure_type,
        }


def load_interface_prompts(path: str | Path = DEFAULT_PROMPTS_PATH) -> list[InterfacePrompt]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Interface robustness prompt file must contain a JSON list")
    prompts = [InterfacePrompt.from_mapping(item) for item in payload]
    seen: set[str] = set()
    for prompt in prompts:
        if prompt.prompt_id in seen:
            raise ValueError(f"Duplicate prompt_id: {prompt.prompt_id}")
        seen.add(prompt.prompt_id)
    return prompts


def run_preflight(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    prompts_path: str | Path = DEFAULT_PROMPTS_PATH,
    reviews_path: str | Path = DEFAULT_REVIEWS_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    run_id: str | None = None,
    model_aliases: Sequence[str] | None = None,
    agent_scopes: Sequence[str] = DEFAULT_AGENT_SCOPES,
    semantic_backend: str = "faiss",
    endpoint_timeout_seconds: float = 3.0,
) -> dict[str, Any]:
    run_dir = _new_run_dir(output_dir, run_id=run_id)
    dotenv_values, runtime_env = _runtime_env()
    selected_labels = _resolve_model_aliases(model_aliases or DEFAULT_PREFLIGHT_MODELS)
    configs = load_model_configs(config_path, selected_labels=selected_labels, env=runtime_env, require_model_ids=False)
    prompts = load_interface_prompts(prompts_path)
    prompt_report = _prompt_schema_report(prompts)
    reviews_valid, review_count = _review_fixture_status(reviews_path)
    scopes = _resolve_agent_scopes(agent_scopes)
    local_endpoint_status = _local_endpoint_status(
        configs,
        runtime_env,
        check_reachability=True,
        endpoint_timeout_seconds=endpoint_timeout_seconds,
    )
    preflight = {
        "phase": "preflight",
        "live_calls_made": False,
        "benchmark_prompts_sent": False,
        "run_dir": str(run_dir),
        "selected_model_labels": selected_labels,
        "selected_model_display_names": [MODEL_DISPLAY_NAMES.get(label, label) for label in selected_labels],
        "agent_scopes": scopes,
        "per_agent_substitution_supported": True,
        "per_agent_substitution_note": (
            "Implemented through provider injection for orchestrator, semantics, and analytics agents; "
            "non-substituted roles use GPT-4o."
        ),
        "prompts_path": str(prompts_path),
        "prompt_count": len(prompts),
        "prompt_schema": prompt_report,
        "reviews_path": str(reviews_path),
        "review_fixture_valid": reviews_valid,
        "review_count": review_count,
        "model_config_status": [_model_status(config, runtime_env) for config in configs],
        "local_endpoint_status": local_endpoint_status,
        "provider_config": _visible_provider_config(runtime_env),
        "semantic_retrieval_backend": semantic_backend,
        "dataset_name": DEFAULT_DATASET_NAME,
        "product_name": DEFAULT_PRODUCT_NAME,
        "status_message": "Preflight complete. No live LLM/model calls were made.",
    }
    _write_output_bundle(
        run_dir,
        rows=[],
        skipped_rows=[],
        manifest=_manifest(
            run_dir=run_dir,
            phase="preflight",
            prompts_path=prompts_path,
            reviews_path=reviews_path,
            prompt_count=len(prompts),
            selected_labels=selected_labels,
            agent_scopes=scopes,
            semantic_backend=semantic_backend,
            live_mode=False,
            extra={"preflight_status": preflight},
        ),
        preflight_status=preflight,
    )
    return preflight


def run_live(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    prompts_path: str | Path = DEFAULT_PROMPTS_PATH,
    reviews_path: str | Path = DEFAULT_REVIEWS_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    run_id: str | None = None,
    model_aliases: Sequence[str],
    agent_scopes: Sequence[str] = DEFAULT_AGENT_SCOPES,
    semantic_backend: str = "faiss",
    max_prompts: int | None = None,
    endpoint_timeout_seconds: float = 3.0,
) -> dict[str, Any]:
    if not model_aliases:
        raise ValueError("--models is required with --run-live")
    run_dir = _new_run_dir(output_dir, run_id=run_id)
    dotenv_values, runtime_env = _runtime_env()
    selected_labels = _resolve_model_aliases(model_aliases)
    scopes = _resolve_agent_scopes(agent_scopes)
    configs = load_model_configs(config_path, selected_labels=selected_labels, env=runtime_env, require_model_ids=True)
    config_by_label = {config.model_label: config for config in configs}
    baseline_config = _baseline_config(config_path, runtime_env)
    prompts = load_interface_prompts(prompts_path)
    if max_prompts is not None:
        prompts = prompts[:max_prompts]
    review_rows = json.loads(Path(reviews_path).read_text(encoding="utf-8"))
    endpoint_status = _local_endpoint_status(
        configs,
        runtime_env,
        check_reachability=True,
        endpoint_timeout_seconds=endpoint_timeout_seconds,
    )
    rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []

    for label in selected_labels:
        config = config_by_label[label]
        endpoint = endpoint_status.get(label)
        if endpoint and not endpoint["reachable"]:
            for scope in scopes:
                skipped_rows.append(_skipped_model_row(config, scope, endpoint["status"]))
            continue
        for scope in scopes:
            mode_name = _mode_name(config, scope)
            settings = _settings_for_run(
                config if scope in {"full", "orchestrator"} else baseline_config,
                dotenv_values,
                runtime_env,
                run_dir=run_dir,
                mode_name=mode_name,
                semantic_backend=semantic_backend,
            )
            ensure_directories(settings)
            role_providers = _build_role_providers(
                scope=scope,
                substituted_config=config,
                baseline_config=baseline_config,
                dotenv_values=dotenv_values,
                runtime_env=runtime_env,
                run_dir=run_dir,
                mode_name=mode_name,
                semantic_backend=semantic_backend,
            )
            with sqlite3.connect(settings.database_path) as conn:
                conn.row_factory = sqlite3.Row
                RetrievalAgent(settings).load_records(conn, DEFAULT_PRODUCT_NAME, review_rows)
                orchestrator = _build_orchestrator(settings, role_providers)
                for prompt in prompts:
                    for provider in role_providers.values():
                        provider.start_prompt(f"{mode_name}:{prompt.prompt_id}")
                    row = _run_interface_prompt(
                        run_dir=run_dir,
                        mode_name=mode_name,
                        config=config,
                        scope=scope,
                        role_providers=role_providers,
                        orchestrator=orchestrator,
                        conn=conn,
                        prompt=prompt,
                    )
                    rows.append(row)

    manifest = _manifest(
        run_dir=run_dir,
        phase="live",
        prompts_path=prompts_path,
        reviews_path=reviews_path,
        prompt_count=len(prompts),
        selected_labels=selected_labels,
        agent_scopes=scopes,
        semantic_backend=semantic_backend,
        live_mode=True,
        extra={
            "local_endpoint_status": endpoint_status,
            "skipped_model_scope_rows": skipped_rows,
            "per_agent_substitution_supported": True,
        },
    )
    output_files = _write_output_bundle(run_dir, rows=rows, skipped_rows=skipped_rows, manifest=manifest)
    return {
        "phase": "live",
        "run_dir": str(run_dir),
        "result_count": len(rows),
        "skipped_model_scope_count": len(skipped_rows),
        "output_files": output_files,
        "live_calls_made": bool(rows),
    }


def run_provider_smoke_test(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    run_id: str | None = None,
    model_aliases: Sequence[str] | None = None,
    semantic_backend: str = "lexical",
) -> dict[str, Any]:
    if not model_aliases:
        raise ValueError("--models is required with --provider-smoke-test")
    run_dir = _new_run_dir(output_dir, run_id=run_id)
    dotenv_values, runtime_env = _runtime_env()
    selected_labels = _resolve_model_aliases(model_aliases)
    local_labels = [label for label in selected_labels if label in LOCAL_MODEL_LABELS]
    if local_labels:
        raise ValueError("--provider-smoke-test is only for cloud providers; do not select Llama or Qwen here.")
    configs = load_model_configs(config_path, selected_labels=selected_labels, env=runtime_env, require_model_ids=True)
    rows = [
        _run_provider_smoke_for_config(
            config,
            dotenv_values,
            runtime_env,
            run_dir=run_dir,
            semantic_backend=semantic_backend,
        )
        for config in configs
    ]
    status = {
        "phase": "provider_smoke_test",
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "live_calls_made": True,
        "selected_model_labels": selected_labels,
        "selected_model_display_names": [MODEL_DISPLAY_NAMES.get(label, label) for label in selected_labels],
        "model_config_status": [_model_status(config, runtime_env) for config in configs],
        "safe_environment": _safe_connection_environment(runtime_env),
        "cloud_proxy_handling": "Proxy environment variables are cleared inside this runner for OpenAI/Anthropic provider calls to avoid inherited dead proxy settings.",
        "provider_results": rows,
        "all_passed": all(row["success"] for row in rows),
        "claim_boundary": _claim_boundary(),
    }
    smoke_path = run_dir / "provider_smoke_status.json"
    smoke_path.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
    manifest = _manifest(
        run_dir=run_dir,
        phase="provider_smoke_test",
        prompts_path=DEFAULT_PROMPTS_PATH,
        reviews_path=DEFAULT_REVIEWS_PATH,
        prompt_count=0,
        selected_labels=selected_labels,
        agent_scopes=(),
        semantic_backend=semantic_backend,
        live_mode=True,
        extra={
            "provider_smoke_status": str(smoke_path),
            "safe_environment": status["safe_environment"],
            "cloud_proxy_handling": status["cloud_proxy_handling"],
        },
    )
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "phase": "provider_smoke_test",
        "run_dir": str(run_dir),
        "provider_smoke_status": str(smoke_path),
        "manifest": str(manifest_path),
        "all_passed": status["all_passed"],
        "live_calls_made": True,
    }


def _run_provider_smoke_for_config(
    config: ModelSubstitutionConfig,
    dotenv_values: Mapping[str, str],
    runtime_env: Mapping[str, str],
    *,
    run_dir: Path,
    semantic_backend: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    settings = _settings_for_run(
        config,
        dotenv_values,
        runtime_env,
        run_dir=run_dir,
        mode_name=f"provider_smoke__{config.model_label}",
        semantic_backend=semantic_backend,
    )
    ensure_directories(settings)
    _install_provider_process_environment(config, _env_for_config(config, dotenv_values, runtime_env, settings=settings))
    result: dict[str, Any] = {
        "model_label": config.model_label,
        "model_display_name": MODEL_DISPLAY_NAMES.get(config.model_label, config.model_label),
        "provider": config.provider,
        "model_id": config.model_id,
        "success": False,
        "exception_type": None,
        "exception_message": None,
        "response_preview": None,
        "latency_seconds": None,
        "token_usage": None,
    }
    try:
        provider = UsageTrackingProvider(
            build_llm_provider(settings),
            provider_name=config.provider,
            model=config.model_id,
        )
        provider.start_prompt(f"provider_smoke:{config.model_label}")
        response = provider.generate(
            "Provider smoke test. Reply with exactly: ok",
            purpose="provider_smoke_test",
        )
        usage = provider.finish_prompt()
        content = str(response.content or "").strip()
        result.update(
            {
                "success": bool(content),
                "response_preview": _redact_sensitive_text(content[:120]),
                "token_usage": usage,
            }
        )
    except Exception as exc:  # pragma: no cover - live provider failures depend on environment/network.
        result.update(
            {
                "exception_type": type(exc).__name__,
                "exception_message": _redact_sensitive_text(str(exc)),
            }
        )
    result["latency_seconds"] = round(time.perf_counter() - started, 3)
    return result


def summarize_only(
    *,
    input_dir: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    run_id: str | None = None,
) -> dict[str, Any]:
    source = Path(input_dir)
    raw_paths = _find_raw_result_files(source)
    if not raw_paths:
        raise FileNotFoundError(f"No raw_results.jsonl files found under {source}")
    run_dir = _new_run_dir(output_dir, prefix="model_interface_robustness_summary", run_id=run_id)
    rows = []
    for raw_path in raw_paths:
        rows.extend(_read_jsonl(raw_path))
    manifest = {
        "run_id": run_dir.name,
        "phase": "summarize_only",
        "source_input_dir": str(source),
        "source_raw_result_files": [str(path) for path in raw_paths],
        "result_count": len(rows),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "code_version": _git_commit_sha(),
        "live_mode": False,
        "live_calls_made": False,
        "claim_boundary": _claim_boundary(),
    }
    output_files = _write_output_bundle(run_dir, rows=rows, skipped_rows=[], manifest=manifest)
    return {
        "phase": "summarize_only",
        "run_dir": str(run_dir),
        "source_raw_result_count": len(raw_paths),
        "result_count": len(rows),
        "output_files": output_files,
        "live_calls_made": False,
    }


def _run_interface_prompt(
    *,
    run_dir: Path,
    mode_name: str,
    config: ModelSubstitutionConfig,
    scope: str,
    role_providers: Mapping[str, UsageTrackingProvider],
    orchestrator: ReviewOrchestrator,
    conn: sqlite3.Connection,
    prompt: InterfacePrompt,
) -> dict[str, Any]:
    started = time.perf_counter()
    result: dict[str, Any] = {}
    trace: dict[str, Any] = {}
    exception_type: str | None = None
    exception_message: str | None = None
    try:
        raw_route = orchestrator.route(prompt.prompt_text)
        if raw_route not in BENCHMARK_MODES["proposed_multi_agent"].supported_routes:
            result = {"type": "error", "message": f"Unsupported route returned by orchestrator: {raw_route}"}
            trace = {"route": raw_route, "table": None, "failure_category": "unsupported_route"}
        else:
            result, trace = _execute_route(orchestrator, conn, prompt.prompt_text, raw_route)
    except Exception as exc:  # pragma: no cover - live provider errors are environment-dependent.
        exception_type = type(exc).__name__
        exception_message = _redact_sensitive_text(str(exc))
        result = {"type": "error", "message": exception_message}
        trace = {"route": None, "table": None, "failure_category": exception_type, "failure_reason": exception_message}
    latency_seconds = round(time.perf_counter() - started, 3)
    usage_by_agent = {
        role: provider.finish_prompt()
        for role, provider in role_providers.items()
    }
    usage = _aggregate_usage(usage_by_agent)
    return _evaluate_row(
        run_dir=run_dir,
        mode_name=mode_name,
        config=config,
        scope=scope,
        prompt=prompt,
        result=result,
        trace=trace,
        latency_seconds=latency_seconds,
        usage=usage,
        usage_by_agent=usage_by_agent,
        exception_type=exception_type,
        exception_message=exception_message,
    )


def _evaluate_row(
    *,
    run_dir: Path,
    mode_name: str,
    config: ModelSubstitutionConfig,
    scope: str,
    prompt: InterfacePrompt,
    result: Mapping[str, Any],
    trace: Mapping[str, Any],
    latency_seconds: float,
    usage: Mapping[str, Any],
    usage_by_agent: Mapping[str, Any],
    exception_type: str | None,
    exception_message: str | None,
) -> dict[str, Any]:
    raw_route = _optional_str(trace.get("route"))
    failure_category = _optional_str(trace.get("failure_category") or result.get("failure_category"))
    failure_reason = _optional_str(trace.get("failure_reason") or result.get("failure_reason") or result.get("message"))
    response_text = _response_text(result)
    evidence_text = " ".join(str(value) for value in trace.get("evidence_snippets", ()) or ()) + " " + response_text
    api_error = bool(exception_type or _usage_has_error(usage))
    timeout = bool(exception_type and "timeout" in exception_type.lower())
    sql_valid = _sql_valid(_optional_str(trace.get("sql")), _optional_str(trace.get("table")))
    chart_spec_valid = _chart_spec_valid(result, trace)
    refusal_correct = _refusal_correct(prompt, failure_category)
    actual_route_for_scoring = "REFUSAL" if prompt.expected_refusal and refusal_correct else raw_route
    routing_correct = actual_route_for_scoring == prompt.expected_route
    structured_output_valid = _structured_output_valid(
        prompt=prompt,
        result=result,
        sql_valid=sql_valid,
        chart_spec_valid=chart_spec_valid,
        refusal_correct=refusal_correct,
        api_error=api_error,
    )
    validation_passed, validation_details = _validation_passed(
        prompt=prompt,
        response_text=response_text,
        evidence_text=evidence_text,
        result=result,
        trace=trace,
        sql_valid=sql_valid,
        chart_spec_valid=chart_spec_valid,
        refusal_correct=refusal_correct,
    )
    task_success = bool(routing_correct and structured_output_valid and validation_passed and not api_error)
    failure_type = _failure_type(
        prompt=prompt,
        task_success=task_success,
        routing_correct=routing_correct,
        sql_valid=sql_valid,
        chart_spec_valid=chart_spec_valid,
        refusal_correct=refusal_correct,
        validation_passed=validation_passed,
        validation_details=validation_details,
        api_error=api_error,
        timeout=timeout,
    )
    return {
        "run_id": run_dir.name,
        "model_setting": mode_name,
        "model_label": config.model_label,
        "model_display_name": MODEL_DISPLAY_NAMES.get(config.model_label, config.model_label),
        "provider": config.provider,
        "model_id": config.model_id,
        "report_model_id": config.report_model_id or config.model_id,
        "agent_scope": scope,
        "substituted_agent": "all" if scope == "full" else scope,
        "prompt_id": prompt.prompt_id,
        "prompt_text": prompt.prompt_text,
        "language": prompt.language,
        "capability_category": prompt.capability_category,
        "expected_route": prompt.expected_route,
        "actual_route": actual_route_for_scoring,
        "raw_route": raw_route,
        "expected_behavior": prompt.expected_behavior,
        "expected_refusal": prompt.expected_refusal,
        "expected_failure_type": prompt.expected_failure_type,
        "controlled_failure_category": failure_category,
        "controlled_failure_reason": failure_reason,
        "task_success": task_success,
        "routing_correct": bool(routing_correct),
        "structured_output_valid": bool(structured_output_valid),
        "sql_valid": sql_valid,
        "chart_spec_valid": chart_spec_valid,
        "refusal_correct": refusal_correct,
        "failure_type": failure_type,
        "validation_rule": prompt.validation_rule,
        "validation_passed": bool(validation_passed),
        "validation_details": validation_details,
        "actual_sql": _optional_str(trace.get("sql")),
        "actual_result_type": _optional_str(result.get("type")),
        "actual_chart_type": _optional_str(result.get("chart_type") or trace.get("chart_type")),
        "actual_chart_grouping": _optional_str(result.get("group_by") or trace.get("chart_group_by")),
        "chart_path": _optional_str(result.get("path") or trace.get("chart_path")),
        "response_preview": response_text[:400],
        "evidence_ids": list(trace.get("evidence_ids", ()) or ()),
        "latency_seconds": latency_seconds,
        "token_input": usage.get("input_tokens"),
        "token_output": usage.get("output_tokens"),
        "token_total": usage.get("total_tokens"),
        "estimated_cost_usd": usage.get("estimated_cost_usd"),
        "token_usage": usage,
        "token_usage_by_agent": usage_by_agent,
        "api_error": api_error,
        "timeout": timeout,
        "exception_type": exception_type,
        "exception_message": exception_message,
        "notes": prompt.notes,
    }


def _validation_passed(
    *,
    prompt: InterfacePrompt,
    response_text: str,
    evidence_text: str,
    result: Mapping[str, Any],
    trace: Mapping[str, Any],
    sql_valid: bool | None,
    chart_spec_valid: bool | None,
    refusal_correct: bool | None,
) -> tuple[bool, str]:
    details: list[str] = []
    passed = True
    if prompt.expected_refusal:
        passed = refusal_correct is True
        return passed, "controlled_refusal" if passed else "expected controlled refusal was not observed"
    if prompt.expected_answer_contains:
        ok = _contains_all(response_text, prompt.expected_answer_contains)
        passed = passed and ok
        details.append("answer_contains=pass" if ok else "answer_contains=fail")
    if prompt.expected_answer_any:
        ok = _contains_any(response_text, prompt.expected_answer_any)
        passed = passed and ok
        details.append("answer_any=pass" if ok else "answer_any=fail")
    if prompt.expected_evidence_contains:
        ok = _contains_all(evidence_text, prompt.expected_evidence_contains)
        passed = passed and ok
        details.append("evidence_contains=pass" if ok else "evidence_contains=fail")
    if prompt.expected_evidence_any:
        ok = _contains_any(evidence_text, prompt.expected_evidence_any)
        passed = passed and ok
        details.append("evidence_any=pass" if ok else "evidence_any=fail")
    if prompt.expected_sql_pattern:
        actual_sql = _optional_str(trace.get("sql")) or ""
        ok = _sql_pattern_matches(prompt, actual_sql)
        passed = passed and ok
        details.append("sql_pattern=pass" if ok else "sql_pattern=fail")
    if prompt.expected_date_range:
        ok = _optional_str(trace.get("date_range")) == prompt.expected_date_range
        passed = passed and ok
        details.append("date_range=pass" if ok else "date_range=fail")
    if prompt.expected_chart_type:
        ok = _optional_str(result.get("chart_type") or trace.get("chart_type")) == prompt.expected_chart_type
        passed = passed and ok
        details.append("chart_type=pass" if ok else "chart_type=fail")
    if prompt.expected_chart_grouping:
        ok = _optional_str(result.get("group_by") or trace.get("chart_group_by")) == prompt.expected_chart_grouping
        passed = passed and ok
        details.append("chart_grouping=pass" if ok else "chart_grouping=fail")
    if prompt.expected_chart_values:
        actual_values = _chart_values(result, trace)
        ok = _numeric_values_match(prompt.expected_chart_values, actual_values)
        passed = passed and ok
        details.append("chart_values=pass" if ok else "chart_values=fail")
    if prompt.expected_route == "DIRECT_SQL":
        passed = passed and sql_valid is True
        details.append("sql_valid=pass" if sql_valid is True else "sql_valid=fail")
    if prompt.validation_rule == "sql_select_only":
        actual_sql = _optional_str(trace.get("sql")) or ""
        ok = _destructive_sql_absent(actual_sql)
        passed = passed and ok
        details.append("destructive_sql_absent=pass" if ok else "destructive_sql_absent=fail")
    if prompt.expected_route == "ANALYTICS":
        passed = passed and chart_spec_valid is True
        details.append("chart_spec_valid=pass" if chart_spec_valid is True else "chart_spec_valid=fail")
    if prompt.validation_rule == "semantic_text_response":
        ok = _optional_str(result.get("type")) == "text" and bool(response_text.strip())
        passed = passed and ok
        details.append("semantic_text_response=pass" if ok else "semantic_text_response=fail")
    return passed, "; ".join(details) if details else "no additional validation fields"


def _structured_output_valid(
    *,
    prompt: InterfacePrompt,
    result: Mapping[str, Any],
    sql_valid: bool | None,
    chart_spec_valid: bool | None,
    refusal_correct: bool | None,
    api_error: bool,
) -> bool:
    if api_error:
        return False
    if prompt.expected_refusal:
        return refusal_correct is True
    if prompt.expected_route == "DIRECT_SQL":
        return sql_valid is True
    if prompt.expected_route == "ANALYTICS":
        return chart_spec_valid is True
    if prompt.expected_route == "SEMANTICS":
        return _optional_str(result.get("type")) == "text"
    return False


def _failure_type(
    *,
    prompt: InterfacePrompt,
    task_success: bool,
    routing_correct: bool,
    sql_valid: bool | None,
    chart_spec_valid: bool | None,
    refusal_correct: bool | None,
    validation_passed: bool,
    validation_details: str,
    api_error: bool,
    timeout: bool,
) -> str:
    if task_success:
        return "none"
    if timeout:
        return "timeout"
    if api_error:
        return "api_error"
    if prompt.expected_refusal:
        return "refusal_error" if refusal_correct is False else "unsupported_overanswer"
    if not routing_correct:
        return "routing_error"
    if prompt.expected_route == "DIRECT_SQL" and sql_valid is not True:
        return "invalid_sql"
    if prompt.expected_route == "ANALYTICS" and chart_spec_valid is not True:
        return "invalid_chart_spec"
    if "evidence_contains=fail" in validation_details:
        return "missing_evidence"
    if not validation_passed:
        return "reasoning_error"
    return "parse_error"


def _build_orchestrator(settings: Settings, role_providers: Mapping[str, UsageTrackingProvider]) -> ReviewOrchestrator:
    orchestrator = ReviewOrchestrator(settings, role_providers["orchestrator"])
    orchestrator.provider = role_providers["orchestrator"]
    orchestrator.semantic_reasoning_agent.provider = role_providers["semantics"]
    orchestrator.analytics_agent.provider = role_providers["analytics"]
    return orchestrator


def _build_role_providers(
    *,
    scope: str,
    substituted_config: ModelSubstitutionConfig,
    baseline_config: ModelSubstitutionConfig,
    dotenv_values: Mapping[str, str],
    runtime_env: Mapping[str, str],
    run_dir: Path,
    mode_name: str,
    semantic_backend: str,
) -> dict[str, UsageTrackingProvider]:
    role_configs = {
        "orchestrator": substituted_config if scope in {"full", "orchestrator"} else baseline_config,
        "semantics": substituted_config if scope in {"full", "semantics"} else baseline_config,
        "analytics": substituted_config if scope in {"full", "analytics"} else baseline_config,
    }
    providers: dict[str, UsageTrackingProvider] = {}
    for role, config in role_configs.items():
        settings = _settings_for_run(
            config,
            dotenv_values,
            runtime_env,
            run_dir=run_dir,
            mode_name=f"{mode_name}__{role}",
            semantic_backend=semantic_backend,
        )
        _install_provider_process_environment(config, _env_for_config(config, dotenv_values, runtime_env, settings=settings))
        wrapped = build_llm_provider(settings)
        providers[role] = UsageTrackingProvider(
            wrapped,
            provider_name=config.provider,
            model=config.model_id,
        )
    return providers


def _settings_for_run(
    config: ModelSubstitutionConfig,
    dotenv_values: Mapping[str, str],
    runtime_env: Mapping[str, str],
    *,
    run_dir: Path,
    mode_name: str,
    semantic_backend: str,
) -> Settings:
    env = dict(dotenv_values)
    env.update(
        {
            "LLM_REVIEW_PROJECT_ROOT": str(run_dir),
            "REVIEWS_DB_PATH": str(run_dir / "runtime" / f"{mode_name}.db"),
            "OUTPUT_DIR": str(run_dir / "charts" / mode_name),
            "VECTORSTORE_DIR": str(run_dir / "vectorstores" / mode_name),
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
        env["HF_ENDPOINT_URL"] = runtime_env[config.endpoint_url_env]
    return load_settings(env)


def _env_for_config(
    config: ModelSubstitutionConfig,
    dotenv_values: Mapping[str, str],
    runtime_env: Mapping[str, str],
    *,
    settings: Settings,
) -> dict[str, str]:
    env = dict(os.environ)
    env.update(dotenv_values)
    env.update(
        {
            "LLM_REVIEW_PROJECT_ROOT": str(settings.project_root),
            "REVIEWS_DB_PATH": str(settings.database_path),
            "OUTPUT_DIR": str(settings.output_dir),
            "VECTORSTORE_DIR": str(settings.vectorstore_dir),
            "LLM_PROVIDER": config.provider,
            "LLM_MODEL": config.model_id,
            "LLM_TEMPERATURE": str(config.temperature),
            "LLM_MAX_TOKENS": str(config.max_tokens),
            "LLM_TIMEOUT_SECONDS": str(config.timeout_seconds),
            "LLM_MAX_RETRIES": str(config.max_retries),
            "SEMANTIC_RETRIEVAL_BACKEND": settings.semantic_retrieval_backend,
            "ALLOW_LIVE_LLM": "true",
            "ALLOW_LIVE_RETRIEVAL": "false",
        }
    )
    if config.endpoint_url_env and runtime_env.get(config.endpoint_url_env):
        env["HF_ENDPOINT_URL"] = runtime_env[config.endpoint_url_env]
    return env


def _write_output_bundle(
    run_dir: Path,
    *,
    rows: list[dict[str, Any]],
    skipped_rows: list[dict[str, Any]],
    manifest: Mapping[str, Any],
    preflight_status: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    raw_results_path = run_dir / "raw_results.jsonl"
    summary_by_model_path = run_dir / "summary_by_model.csv"
    summary_by_category_path = run_dir / "summary_by_model_and_category.csv"
    failure_analysis_path = run_dir / "failure_analysis.csv"
    report_path = run_dir / "model_substitution_interface_robustness_report.md"
    manifest_path = run_dir / "manifest.json"
    preflight_path = run_dir / "preflight_status.json"
    _write_jsonl(raw_results_path, rows)
    summary_rows = _summary_by_model(rows, skipped_rows)
    category_rows = _summary_by_model_and_category(rows)
    failure_rows = _failure_analysis(rows)
    _write_csv(summary_by_model_path, summary_rows, _summary_by_model_fields())
    _write_csv(summary_by_category_path, category_rows, _summary_by_category_fields())
    _write_csv(failure_analysis_path, failure_rows, _failure_analysis_fields())
    report_path.write_text(
        _markdown_report(
            rows=rows,
            skipped_rows=skipped_rows,
            summary_rows=summary_rows,
            category_rows=category_rows,
            failure_rows=failure_rows,
            manifest=manifest,
            preflight_status=preflight_status,
        ),
        encoding="utf-8",
    )
    _write_json(manifest_path, dict(manifest))
    output_files = {
        "raw_results": str(raw_results_path),
        "summary_by_model": str(summary_by_model_path),
        "summary_by_model_and_category": str(summary_by_category_path),
        "failure_analysis": str(failure_analysis_path),
        "report": str(report_path),
        "manifest": str(manifest_path),
    }
    if preflight_status is not None:
        _write_json(preflight_path, dict(preflight_status))
        output_files["preflight_status"] = str(preflight_path)
    return output_files


def _summary_by_model(rows: Sequence[Mapping[str, Any]], skipped_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["model_setting"])].append(row)
    summary: list[dict[str, Any]] = []
    for model_setting, group in sorted(grouped.items()):
        summary.append(_summary_row(model_setting, group))
    summary.extend(dict(row) for row in skipped_rows)
    return summary


def _summary_row(model_setting: str, group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    first = group[0]
    token_total = _sum_optional(row.get("token_total") for row in group)
    input_total = _sum_optional(row.get("token_input") for row in group)
    output_total = _sum_optional(row.get("token_output") for row in group)
    cost_total = _sum_optional(row.get("estimated_cost_usd") for row in group)
    return {
        "model_setting": model_setting,
        "model_label": first.get("model_label"),
        "model_display_name": first.get("model_display_name"),
        "provider": first.get("provider"),
        "model_id": first.get("model_id"),
        "agent_scope": first.get("agent_scope"),
        "run_status": "completed",
        "prompt_count": len(group),
        "task_success_rate": _bool_rate(row.get("task_success") for row in group),
        "routing_accuracy": _bool_rate(row.get("routing_correct") for row in group),
        "structured_output_valid_rate": _bool_rate(row.get("structured_output_valid") for row in group),
        "sql_valid_rate": _bool_rate(row.get("sql_valid") for row in group if row.get("sql_valid") is not None),
        "chart_spec_valid_rate": _bool_rate(row.get("chart_spec_valid") for row in group if row.get("chart_spec_valid") is not None),
        "refusal_correct_rate": _bool_rate(row.get("refusal_correct") for row in group if row.get("refusal_correct") is not None),
        "api_error_count": sum(1 for row in group if row.get("api_error")),
        "timeout_count": sum(1 for row in group if row.get("timeout")),
        "avg_latency_seconds": _avg(row.get("latency_seconds") for row in group),
        "token_input": input_total,
        "token_output": output_total,
        "token_total": token_total,
        "estimated_cost_usd": cost_total,
        "failure_types": json.dumps(dict(sorted(Counter(str(row.get("failure_type", "none")) for row in group).items())), sort_keys=True),
    }


def _summary_by_model_and_category(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model_setting"]), str(row["capability_category"]))].append(row)
    summary: list[dict[str, Any]] = []
    for (model_setting, category), group in sorted(grouped.items()):
        first = group[0]
        summary.append(
            {
                "model_setting": model_setting,
                "model_display_name": first.get("model_display_name"),
                "agent_scope": first.get("agent_scope"),
                "capability_category": category,
                "prompt_count": len(group),
                "task_success_rate": _bool_rate(row.get("task_success") for row in group),
                "routing_accuracy": _bool_rate(row.get("routing_correct") for row in group),
                "structured_output_valid_rate": _bool_rate(row.get("structured_output_valid") for row in group),
                "refusal_correct_rate": _bool_rate(row.get("refusal_correct") for row in group if row.get("refusal_correct") is not None),
                "avg_latency_seconds": _avg(row.get("latency_seconds") for row in group),
                "failure_types": json.dumps(dict(sorted(Counter(str(row.get("failure_type", "none")) for row in group).items())), sort_keys=True),
            }
        )
    return summary


def _failure_analysis(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        failure_type = str(row.get("failure_type") or "none")
        if failure_type == "none":
            continue
        grouped[(str(row["model_setting"]), failure_type)].append(row)
    failure_rows: list[dict[str, Any]] = []
    for (model_setting, failure_type), group in sorted(grouped.items()):
        first = group[0]
        failure_rows.append(
            {
                "model_setting": model_setting,
                "model_display_name": first.get("model_display_name"),
                "agent_scope": first.get("agent_scope"),
                "failure_type": failure_type,
                "count": len(group),
                "prompt_ids": ";".join(str(row["prompt_id"]) for row in group),
                "capability_categories": ";".join(sorted({str(row["capability_category"]) for row in group})),
                "example_reason": _optional_str(group[0].get("validation_details") or group[0].get("controlled_failure_reason")) or "",
            }
        )
    return failure_rows


def _markdown_report(
    *,
    rows: Sequence[Mapping[str, Any]],
    skipped_rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
    category_rows: Sequence[Mapping[str, Any]],
    failure_rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    preflight_status: Mapping[str, Any] | None,
) -> str:
    prompt_count = manifest.get("prompt_count", 0)
    model_settings = sorted({str(row.get("model_setting")) for row in rows})
    if not model_settings:
        model_settings = [str(row.get("model_setting")) for row in skipped_rows]
    if not model_settings and preflight_status is not None:
        model_settings = [str(value) for value in preflight_status.get("selected_model_display_names", ())]
    lines = [
        "# Model-Substitution and Interface-Robustness Analysis",
        "",
        "## Purpose and Scope",
        "",
        "This analysis tests sensitivity of the review-analysis workflow when LLM providers are substituted. It measures interface behavior across routing, structured outputs, controlled refusals, latency, and token metadata. It is not a model leaderboard, a full LLM benchmark, or proof of complete model agnosticism.",
        "",
        "## Dataset and Prompt Set",
        "",
        f"- Dataset: {manifest.get('dataset_name', DEFAULT_DATASET_NAME)}.",
        f"- Product/table target: {manifest.get('product_name', DEFAULT_PRODUCT_NAME)}.",
        f"- Prompt file: `{manifest.get('prompts_path', DEFAULT_PROMPTS_PATH)}`.",
        f"- Prompt count: {prompt_count}.",
        "- Prompt groups: Direct SQL/factual queries, date/rating/count queries, semantic evidence questions, analytics/chart requests, multilingual prompts, and refusal-boundary prompts.",
        "",
        "## Model Configurations",
        "",
        f"- Model settings executed or checked: {', '.join(model_settings) if model_settings else 'none'}.",
        f"- Agent scopes: {', '.join(str(value) for value in manifest.get('agent_scopes', [])) or 'not applicable'}.",
        "- Per-agent substitution: supported through provider injection for the Orchestrator, Semantics Agent, and Data Analytics Agent. Non-substituted roles use GPT-4o.",
        "- Local Qwen/Llama endpoints are checked only when selected; this runner does not start, stop, pause, or kill local model processes.",
        "",
        "## Metrics",
        "",
        "The raw JSONL records include task_success, routing_correct, structured_output_valid, sql_valid, chart_spec_valid, refusal_correct, failure_type, latency_seconds, token_input, token_output, token_total, estimated_cost_usd, api_error, and timeout. Token fields are null when provider metadata is unavailable. Dollar cost remains null unless reliable per-provider pricing is configured externally.",
        "",
        "## Results Table",
        "",
        _markdown_table(summary_rows, ("model_setting", "prompt_count", "task_success_rate", "routing_accuracy", "structured_output_valid_rate", "refusal_correct_rate", "avg_latency_seconds", "token_total", "run_status")),
        "",
        "## Category-Level Table",
        "",
        _markdown_table(category_rows, ("model_setting", "capability_category", "prompt_count", "task_success_rate", "routing_accuracy", "structured_output_valid_rate", "refusal_correct_rate")),
        "",
        "## Failure-Type Summary",
        "",
        _markdown_table(failure_rows, ("model_setting", "failure_type", "count", "prompt_ids")) if failure_rows else "No non-success failure types were recorded in the available rows.",
        "",
        "## Short Interpretation",
        "",
        _interpretation(rows, skipped_rows, preflight_status),
        "",
        "## Limitations",
        "",
        "- This is a bounded cross-model workflow evaluation over a controlled 250-review artifact.",
        "- The prompt set is broader than the earlier pilot but remains small and task-oriented.",
        "- Programmatic checks validate workflow behavior and structured outputs; they are not a substitute for human evaluation of nuanced answer quality.",
        "- Local endpoint availability, provider API behavior, and token metadata can vary by deployment.",
        "- Cost estimates remain null unless authoritative current pricing is configured for the selected providers.",
    ]
    return "\n".join(lines) + "\n"


def _interpretation(
    rows: Sequence[Mapping[str, Any]],
    skipped_rows: Sequence[Mapping[str, Any]],
    preflight_status: Mapping[str, Any] | None,
) -> str:
    if preflight_status is not None and not rows:
        return "Preflight completed without sending benchmark prompts or making live LLM/model calls. The output bundle is a readiness artifact, not an empirical result."
    if not rows:
        if skipped_rows:
            return "No prompts were executed because selected model endpoints were unavailable or skipped. These rows are recorded as not run, not failed."
        return "No prompt execution rows were available for interpretation."
    success_rate = _bool_rate(row.get("task_success") for row in rows)
    routing_rate = _bool_rate(row.get("routing_correct") for row in rows)
    return (
        f"Across the available execution rows, task success was {success_rate} and routing correctness was {routing_rate}. "
        "These values should be read as workflow sensitivity indicators for the selected configurations, not as broad comparative model quality claims."
    )


def _manifest(
    *,
    run_dir: Path,
    phase: str,
    prompts_path: str | Path,
    reviews_path: str | Path,
    prompt_count: int,
    selected_labels: Sequence[str],
    agent_scopes: Sequence[str],
    semantic_backend: str,
    live_mode: bool,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        "run_id": run_dir.name,
        "phase": phase,
        "mode": "model_substitution_interface_robustness",
        "dataset_name": DEFAULT_DATASET_NAME,
        "product_name": DEFAULT_PRODUCT_NAME,
        "prompts_path": str(prompts_path),
        "reviews_path": str(reviews_path),
        "output_dir": str(run_dir),
        "prompt_count": prompt_count,
        "selected_model_labels": list(selected_labels),
        "agent_scopes": list(agent_scopes),
        "semantic_retrieval_backend": semantic_backend,
        "live_mode": live_mode,
        "live_calls_made": live_mode,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "code_version": _git_commit_sha(),
        "claim_boundary": _claim_boundary(),
        "local_model_process_control": "The runner never starts, stops, pauses, or kills Qwen/Llama local model processes.",
    }
    if extra:
        manifest.update(extra)
    return manifest


def _claim_boundary() -> str:
    return (
        "Cross-model workflow evaluation only; "
        "not a model leaderboard, not a full LLM benchmark, and not proof of model agnosticism."
    )


def _prompt_schema_report(prompts: Sequence[InterfacePrompt]) -> dict[str, Any]:
    categories = Counter(prompt.capability_category for prompt in prompts)
    routes = Counter(prompt.expected_route for prompt in prompts)
    validation_rules = Counter(prompt.validation_rule for prompt in prompts)
    required_categories = {
        "direct_sql_factual",
        "date_rating_count_queries",
        "semantic_reasoning_evidence",
        "analytics_chart_specification",
        "multilingual_prompts",
        "refusal_boundary",
    }
    return {
        "valid": True,
        "prompt_count": len(prompts),
        "approximately_25_to_30": 25 <= len(prompts) <= 30,
        "categories": dict(sorted(categories.items())),
        "expected_routes": dict(sorted(routes.items())),
        "validation_rules": dict(sorted(validation_rules.items())),
        "required_categories_present": sorted(required_categories.intersection(categories)),
        "missing_required_categories": sorted(required_categories.difference(categories)),
        "expected_refusal_count": sum(1 for prompt in prompts if prompt.expected_refusal),
    }


def _model_status(config: ModelSubstitutionConfig, env: Mapping[str, str]) -> dict[str, Any]:
    key_names = config.required_key_names()
    return {
        "model_label": config.model_label,
        "display_name": MODEL_DISPLAY_NAMES.get(config.model_label, config.model_label),
        "provider": config.provider,
        "model_id": config.model_id if not config.model_id.startswith("${") else "<unresolved>",
        "report_model_id": config.report_model_id if not (config.report_model_id or "").startswith("${") else "<unresolved>",
        "api_key_present": any(bool(env.get(key)) for key in key_names),
        "api_key_names": list(key_names),
        "endpoint_url_env": config.endpoint_url_env,
        "endpoint_url_present": bool(config.endpoint_url_env and env.get(config.endpoint_url_env)),
        "local_model_requires_manual_process": config.model_label in LOCAL_MODEL_LABELS,
    }


def _local_endpoint_status(
    configs: Sequence[ModelSubstitutionConfig],
    env: Mapping[str, str],
    *,
    check_reachability: bool,
    endpoint_timeout_seconds: float,
) -> dict[str, dict[str, Any]]:
    status: dict[str, dict[str, Any]] = {}
    for config in configs:
        if config.model_label not in LOCAL_MODEL_LABELS:
            continue
        endpoint_env = config.endpoint_url_env
        endpoint = env.get(endpoint_env or "", "")
        if not endpoint:
            status[config.model_label] = {
                "endpoint_url_env": endpoint_env,
                "configured": False,
                "reachable": False,
                "status": "missing_endpoint_url",
                "note": "Start the local model manually and set the endpoint URL before selecting this model.",
            }
            continue
        if not check_reachability:
            status[config.model_label] = {
                "endpoint_url_env": endpoint_env,
                "configured": True,
                "reachable": None,
                "status": "not_checked",
                "note": "Endpoint reachability was not requested.",
            }
            continue
        reachable, detail = _endpoint_reachable(endpoint, timeout_seconds=endpoint_timeout_seconds)
        status[config.model_label] = {
            "endpoint_url_env": endpoint_env,
            "configured": True,
            "reachable": reachable,
            "status": "reachable" if reachable else "endpoint_unreachable",
            "detail": detail,
            "note": "The runner does not start or stop the local model process.",
        }
    return status


def _endpoint_reachable(url: str, *, timeout_seconds: float) -> tuple[bool, str]:
    try:
        req = urllib_request.Request(url, method="GET")
        with urllib_request.urlopen(req, timeout=timeout_seconds):
            return True, "GET succeeded"
    except urllib_error.HTTPError as exc:
        return True, f"HTTP {exc.code} response indicates a reachable server"
    except Exception as exc:
        return False, _redact_sensitive_text(str(exc))


def _skipped_model_row(config: ModelSubstitutionConfig, scope: str, status: str) -> dict[str, Any]:
    return {
        "model_setting": _mode_name(config, scope),
        "model_label": config.model_label,
        "model_display_name": MODEL_DISPLAY_NAMES.get(config.model_label, config.model_label),
        "provider": config.provider,
        "model_id": config.model_id,
        "agent_scope": scope,
        "run_status": status,
        "prompt_count": 0,
        "task_success_rate": None,
        "routing_accuracy": None,
        "structured_output_valid_rate": None,
        "sql_valid_rate": None,
        "chart_spec_valid_rate": None,
        "refusal_correct_rate": None,
        "api_error_count": 0,
        "timeout_count": 0,
        "avg_latency_seconds": None,
        "token_input": None,
        "token_output": None,
        "token_total": None,
        "estimated_cost_usd": None,
        "failure_types": "{}",
    }


def _baseline_config(config_path: str | Path, env: Mapping[str, str]) -> ModelSubstitutionConfig:
    return load_model_configs(config_path, selected_labels=["gpt4o_primary"], env=env, require_model_ids=True)[0]


def _mode_name(config: ModelSubstitutionConfig, scope: str) -> str:
    if scope == "full":
        return f"full__{config.model_label}"
    return f"{scope}_only__{config.model_label}"


def _resolve_model_aliases(aliases: Sequence[str]) -> list[str]:
    labels: list[str] = []
    for alias in aliases:
        key = alias.strip().lower()
        if not key:
            continue
        if key not in MODEL_ALIASES:
            raise ValueError(f"Unknown model alias '{alias}'. Supported aliases: {', '.join(sorted(MODEL_ALIASES))}")
        label = MODEL_ALIASES[key]
        if label not in labels:
            labels.append(label)
    return labels


def _resolve_agent_scopes(scopes: Sequence[str]) -> list[str]:
    resolved: list[str] = []
    for scope in scopes:
        normalized = scope.strip().lower()
        if normalized == "all":
            normalized_scopes = ("full", "orchestrator", "semantics", "analytics")
        else:
            normalized_scopes = (normalized,)
        for item in normalized_scopes:
            if item not in AGENT_SCOPES:
                raise ValueError(f"Unknown agent scope '{scope}'. Supported scopes: {', '.join(sorted(AGENT_SCOPES))}, all")
            if item not in resolved:
                resolved.append(item)
    return resolved


def _runtime_env() -> tuple[dict[str, str], dict[str, str]]:
    dotenv_values = read_dotenv(PROJECT_ROOT / ".env")
    runtime_env = dict(os.environ)
    runtime_env.update(dotenv_values)
    return dotenv_values, runtime_env


def _install_provider_process_environment(config: ModelSubstitutionConfig, values: Mapping[str, str]) -> None:
    _install_process_environment(values)
    if config.provider in CLOUD_PROVIDER_NAMES:
        for key in PROXY_ENV_KEYS:
            os.environ.pop(key, None)


def _new_run_dir(
    output_dir: str | Path,
    *,
    prefix: str = "model_interface_robustness",
    run_id: str | None = None,
) -> Path:
    root = Path(output_dir)
    run_name = run_id or f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:6]}"
    run_dir = root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _review_fixture_status(path: str | Path) -> tuple[bool, int | None]:
    fixture_path = Path(path)
    if not fixture_path.exists():
        return False, None
    try:
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    except Exception:
        return False, None
    return isinstance(payload, list), len(payload) if isinstance(payload, list) else None


def _visible_provider_config(values: Mapping[str, str]) -> dict[str, Any]:
    return _scrubbed_provider_config(
        {
            key: str(values[key])
            for key in TRACKED_PROVIDER_CONFIG_KEYS
            if key in values and values[key] is not None
        }
    )


def _safe_connection_environment(values: Mapping[str, str]) -> dict[str, Any]:
    key_names = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN")
    return {
        "api_keys": {
            key: {
                "set": bool(values.get(key)),
                "length": len(values[key]) if values.get(key) else 0,
            }
            for key in key_names
        },
        "base_urls": _safe_value_status(values, BASE_URL_ENV_KEYS),
        "proxy_variables": {
            key: {
                "set": bool(values.get(key)),
                "value": values.get(key) if values.get(key) else None,
                "looks_like_loopback_blackhole": _looks_like_loopback_blackhole(values.get(key)),
            }
            for key in PROXY_ENV_KEYS
        },
        "project_endpoint_variables": _safe_value_status(values, PROJECT_ENDPOINT_ENV_KEYS),
        "runtime": _safe_value_status(
            values,
            (
                "LLM_PROVIDER",
                "LLM_MODEL",
                "ANTHROPIC_MODEL_ID",
                "LLM_TIMEOUT_SECONDS",
                "LLM_MAX_RETRIES",
            ),
        ),
    }


def _safe_value_status(values: Mapping[str, str], keys: Sequence[str]) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "set": bool(values.get(key)),
            "value": values.get(key) if values.get(key) else None,
        }
        for key in keys
    }


def _looks_like_loopback_blackhole(value: str | None) -> bool:
    if not value:
        return False
    lowered = value.lower()
    return ("127.0.0.1:9" in lowered) or ("localhost:9" in lowered) or ("[::1]:9" in lowered)


def _sql_valid(sql: str | None, table: str | None) -> bool | None:
    if not sql:
        return None
    if not table:
        return False
    try:
        validate_select_sql(sql, allowed_tables=[table], allowed_columns=REVIEW_COLUMNS)
    except SQLValidationError:
        return False
    return True


def _chart_spec_valid(result: Mapping[str, Any], trace: Mapping[str, Any]) -> bool | None:
    chart_type = _optional_str(result.get("chart_type") or trace.get("chart_type"))
    if not chart_type and _optional_str(result.get("type")) != "chart":
        return None
    group_by = _optional_str(result.get("group_by") or trace.get("chart_group_by"))
    chart_path = _optional_str(result.get("path") or trace.get("chart_path"))
    if result.get("failure_category") or trace.get("failure_category"):
        return False
    if chart_type not in CHART_TYPES:
        return False
    if group_by and group_by not in REVIEW_COLUMNS:
        return False
    return bool(chart_path and Path(chart_path).exists())


def _refusal_correct(prompt: InterfacePrompt, failure_category: str | None) -> bool | None:
    if not prompt.expected_refusal:
        return None
    if not failure_category:
        return False
    if prompt.expected_failure_type:
        return failure_category == prompt.expected_failure_type
    return True


def _response_text(result: Mapping[str, Any]) -> str:
    values = [
        result.get("message"),
        result.get("explanation"),
        result.get("text"),
    ]
    return " ".join(str(value) for value in values if value)


def _contains_all(text: str, needles: Sequence[str]) -> bool:
    lower = text.lower()
    return all(str(needle).lower() in lower for needle in needles)


def _contains_any(text: str, needles: Sequence[str]) -> bool:
    if not needles:
        return True
    lower = text.lower()
    return any(str(needle).lower() in lower for needle in needles)


def _sql_pattern_matches(prompt: InterfacePrompt, actual_sql: str) -> bool:
    if prompt.expected_sql_pattern and re.search(prompt.expected_sql_pattern, actual_sql, flags=re.IGNORECASE | re.DOTALL):
        return True
    if not prompt.expected_date_range:
        return False
    if not _date_filter_sql_matches(actual_sql, prompt.expected_date_range):
        return False
    if prompt.validation_rule == "date_range_and_count":
        return bool(re.search(r"\bCOUNT\s*\(\s*\*\s*\)", actual_sql, flags=re.IGNORECASE))
    if prompt.validation_rule == "date_range_and_average":
        return bool(re.search(r"\bAVG\s*\(", actual_sql, flags=re.IGNORECASE))
    return True


def _date_filter_sql_matches(actual_sql: str, expected_date_range: str) -> bool:
    if ".." not in expected_date_range:
        return False
    start, end = (part.strip() for part in expected_date_range.split("..", 1))
    if not start or not end:
        return False
    quote = r"['\"]?"
    optional_open = r"\(*\s*"
    optional_close = r"\s*\)*"
    ge_le = (
        rf"{optional_open}date\s*>=\s*{quote}{re.escape(start)}{quote}{optional_close}"
        rf"\s+AND\s+"
        rf"{optional_open}date\s*<=\s*{quote}{re.escape(end)}{quote}{optional_close}"
    )
    le_ge = (
        rf"{optional_open}date\s*<=\s*{quote}{re.escape(end)}{quote}{optional_close}"
        rf"\s+AND\s+"
        rf"{optional_open}date\s*>=\s*{quote}{re.escape(start)}{quote}{optional_close}"
    )
    between = rf"{optional_open}date\s+BETWEEN\s+{quote}{re.escape(start)}{quote}\s+AND\s+{quote}{re.escape(end)}{quote}{optional_close}"
    return any(re.search(pattern, actual_sql, flags=re.IGNORECASE | re.DOTALL) for pattern in (ge_le, le_ge, between))


def _destructive_sql_absent(actual_sql: str) -> bool:
    return not re.search(r"\b(?:DROP|DELETE|UPDATE|INSERT|ALTER|CREATE|TRUNCATE|REPLACE)\b", actual_sql, flags=re.IGNORECASE)


def _chart_values(result: Mapping[str, Any], trace: Mapping[str, Any]) -> dict[str, float]:
    rows = result.get("chart_rows") or trace.get("chart_rows") or []
    values: dict[str, float] = {}
    for row in rows:
        if isinstance(row, Mapping):
            label = str(row.get("label"))
            raw_value = row.get("value")
        else:
            try:
                label = str(row[0])
                raw_value = row[1]
            except Exception:
                continue
        try:
            values[label] = float(raw_value)
        except (TypeError, ValueError):
            continue
    return values


def _numeric_values_match(expected: Mapping[str, float], actual: Mapping[str, float], *, tolerance: float = 1e-6) -> bool:
    if set(expected) != set(actual):
        return False
    return all(abs(float(expected[key]) - float(actual[key])) <= tolerance for key in expected)


def _aggregate_usage(usage_by_agent: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []
    input_tokens = output_tokens = total_tokens = 0
    has_input = has_output = has_total = False
    for role, usage in usage_by_agent.items():
        for call in usage.get("calls", ()) or ():
            call_payload = dict(call)
            call_payload["agent_role"] = role
            calls.append(call_payload)
        if isinstance(usage.get("input_tokens"), (int, float)):
            has_input = True
            input_tokens += int(usage["input_tokens"])
        if isinstance(usage.get("output_tokens"), (int, float)):
            has_output = True
            output_tokens += int(usage["output_tokens"])
        if isinstance(usage.get("total_tokens"), (int, float)):
            has_total = True
            total_tokens += int(usage["total_tokens"])
    return {
        "input_tokens": input_tokens if has_input else None,
        "output_tokens": output_tokens if has_output else None,
        "total_tokens": total_tokens if has_total else None,
        "call_count": len(calls),
        "calls": calls,
        "estimated_cost_usd": None,
        "cost_note": "Cost is null unless authoritative current provider pricing is configured externally.",
    }


def _usage_has_error(usage: Mapping[str, Any]) -> bool:
    return any(str(call.get("status")) == "error" for call in usage.get("calls", ()) or ())


def _find_raw_result_files(source: Path) -> list[Path]:
    if source.is_file() and source.name == "raw_results.jsonl":
        return [source]
    if source.is_dir() and (source / "raw_results.jsonl").exists():
        return [source / "raw_results.jsonl"]
    if source.is_dir():
        return sorted(source.rglob("raw_results.jsonl"))
    return []


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fieldnames})


def _summary_by_model_fields() -> list[str]:
    return [
        "model_setting",
        "model_label",
        "model_display_name",
        "provider",
        "model_id",
        "agent_scope",
        "run_status",
        "prompt_count",
        "task_success_rate",
        "routing_accuracy",
        "structured_output_valid_rate",
        "sql_valid_rate",
        "chart_spec_valid_rate",
        "refusal_correct_rate",
        "api_error_count",
        "timeout_count",
        "avg_latency_seconds",
        "token_input",
        "token_output",
        "token_total",
        "estimated_cost_usd",
        "failure_types",
    ]


def _summary_by_category_fields() -> list[str]:
    return [
        "model_setting",
        "model_display_name",
        "agent_scope",
        "capability_category",
        "prompt_count",
        "task_success_rate",
        "routing_accuracy",
        "structured_output_valid_rate",
        "refusal_correct_rate",
        "avg_latency_seconds",
        "failure_types",
    ]


def _failure_analysis_fields() -> list[str]:
    return [
        "model_setting",
        "model_display_name",
        "agent_scope",
        "failure_type",
        "count",
        "prompt_ids",
        "capability_categories",
        "example_reason",
    ]


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def _markdown_table(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    if not rows:
        return "No rows available."
    header = "| " + " | ".join(fields) + " |"
    separator = "| " + " | ".join("---" for _ in fields) + " |"
    body = [
        "| " + " | ".join(_markdown_cell(row.get(field)) for field in fields) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def _markdown_cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _bool_rate(values: Sequence[Any] | Any) -> float | None:
    value_list = [value for value in values if value is not None]
    if not value_list:
        return None
    return round(sum(1 for value in value_list if bool(value)) / len(value_list), 4)


def _avg(values: Sequence[Any] | Any) -> float | None:
    value_list = [float(value) for value in values if isinstance(value, (int, float))]
    if not value_list:
        return None
    return round(sum(value_list) / len(value_list), 4)


def _sum_optional(values: Sequence[Any] | Any) -> float | int | None:
    value_list = [value for value in values if isinstance(value, (int, float))]
    if not value_list:
        return None
    total = sum(value_list)
    return int(total) if float(total).is_integer() else round(float(total), 6)


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
    redacted = re.sub(r"\b(?:sk|hf)-[A-Za-z0-9_\-]{20,}\b", "<redacted secret>", redacted)
    return redacted


def _print_local_model_instructions(model_labels: Sequence[str]) -> None:
    local = [label for label in model_labels if label in LOCAL_MODEL_LABELS]
    if not local:
        return
    print("Local model note:", file=sys.stderr)
    for label in local:
        endpoint_name = "HF_LLAMA_ENDPOINT_URL" if "llama" in label else "HF_QWEN_ENDPOINT_URL"
        print(
            f"- {MODEL_DISPLAY_NAMES.get(label, label)} was selected. Start it manually, set {endpoint_name}, "
            "run this command, then stop/pause it manually when finished. This script will not manage the process.",
            file=sys.stderr,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-model workflow evaluation runner.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS_PATH)
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Model aliases. Defaults to the manuscript-aligned set: gpt4o, claude, llama, qwen.",
    )
    parser.add_argument("--agent-scopes", nargs="*", default=list(DEFAULT_AGENT_SCOPES), help="Scopes: full, orchestrator, semantics, analytics, all.")
    parser.add_argument("--semantic-backend", default="faiss", choices=("faiss", "lexical"))
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--endpoint-timeout-seconds", type=float, default=3.0)
    parser.add_argument("--preflight", action="store_true", help="Check files, schema, env, and selected local endpoints without live LLM calls.")
    parser.add_argument("--provider-smoke-test", action="store_true", help="Make one tiny approved live call per selected cloud provider and write provider_smoke_status.json.")
    parser.add_argument("--run-live", action="store_true", help="Required for any live API/model calls.")
    parser.add_argument("--summarize-only", action="store_true", help="Merge/summarize existing raw_results.jsonl files without live calls.")
    parser.add_argument("--input-dir", type=Path, default=None, help="Input directory for --summarize-only.")
    args = parser.parse_args()

    if args.summarize_only:
        if args.input_dir is None:
            parser.error("--input-dir is required with --summarize-only")
        result = summarize_only(input_dir=args.input_dir, output_dir=args.output_dir, run_id=args.run_id)
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    selected_labels = _resolve_model_aliases(args.models or DEFAULT_PREFLIGHT_MODELS)
    _print_local_model_instructions(selected_labels)

    if args.preflight:
        result = run_preflight(
            config_path=args.config,
            prompts_path=args.prompts,
            reviews_path=args.reviews,
            output_dir=args.output_dir,
            run_id=args.run_id,
            model_aliases=args.models,
            agent_scopes=args.agent_scopes,
            semantic_backend=args.semantic_backend,
            endpoint_timeout_seconds=args.endpoint_timeout_seconds,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if args.provider_smoke_test:
        if not args.run_live:
            parser.error("--run-live is required with --provider-smoke-test")
        if not args.models:
            parser.error("--models is required with --provider-smoke-test")
        result = run_provider_smoke_test(
            config_path=args.config,
            output_dir=args.output_dir,
            run_id=args.run_id,
            model_aliases=args.models,
            semantic_backend=args.semantic_backend,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if not args.run_live:
        parser.error("--run-live is required for prompt execution. Use --preflight for no-call readiness checks.")

    if not args.models:
        parser.error("--models is required with --run-live")

    result = run_live(
        config_path=args.config,
        prompts_path=args.prompts,
        reviews_path=args.reviews,
        output_dir=args.output_dir,
        run_id=args.run_id,
        model_aliases=args.models,
        agent_scopes=args.agent_scopes,
        semantic_backend=args.semantic_backend,
        max_prompts=args.max_prompts,
        endpoint_timeout_seconds=args.endpoint_timeout_seconds,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
