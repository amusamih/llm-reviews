from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import subprocess
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
from evaluation.run_benchmark import (
    BENCHMARK_EVALUATION_TAGS,
    BENCHMARK_MODES,
    _run_prompt,
    _write_json,
    _write_jsonl,
    load_gold_benchmark_prompts,
)
from llm_review_analysis.agents import RetrievalAgent, ReviewOrchestrator
from llm_review_analysis.config import ensure_directories, load_settings
from llm_review_analysis.llm import LLMProvider, LLMResponse
from llm_review_analysis.providers import build_llm_provider


DEFAULT_PROMPTS_PATH = PROJECT_ROOT / "outputs" / "programmatic_gold" / "amazon_all_beauty_20260624" / "programmatic_gold_prompts.json"
DEFAULT_REVIEWS_PATH = PROJECT_ROOT / "outputs" / "programmatic_gold" / "amazon_all_beauty_20260624" / "programmatic_gold_reviews.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "benchmarks"
DEFAULT_EVIDENCE_ID = "EVID-LIVE-GPT4O-PILOT-001"
DEFAULT_PROVIDER = "langchain"
DEFAULT_MODEL = "gpt-4o"
SECRET_KEY_NAMES = {
    "ANTHROPIC_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "HF_ENDPOINT_URL",
    "HF_LLAMA_ENDPOINT_URL",
    "HF_QWEN_ENDPOINT_URL",
    "HF_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
    "OPENAI_API_KEY",
}


class UsageTrackingProvider:
    def __init__(self, wrapped: LLMProvider, *, provider_name: str, model: str) -> None:
        self.wrapped = wrapped
        self.provider_name = provider_name
        self.model = model
        self._current_prompt_id: str | None = None
        self._current_calls: list[dict[str, Any]] = []

    def start_prompt(self, prompt_id: str) -> None:
        self._current_prompt_id = prompt_id
        self._current_calls = []

    def finish_prompt(self) -> dict[str, Any]:
        usage = _aggregate_call_usage(self._current_calls)
        usage["calls"] = list(self._current_calls)
        return usage

    def generate(self, prompt: str, *, purpose: str = "general", response_format: str | None = None) -> LLMResponse:
        started_at = datetime.now(timezone.utc).isoformat()
        try:
            response = self.wrapped.generate(prompt, purpose=purpose, response_format=response_format)
        except Exception as exc:
            self._current_calls.append(
                {
                    "prompt_id": self._current_prompt_id,
                    "purpose": purpose,
                    "response_format": response_format,
                    "started_at_utc": started_at,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "usage": {},
                }
            )
            raise
        usage = dict(response.usage or {})
        self._current_calls.append(
            {
                "prompt_id": self._current_prompt_id,
                "purpose": purpose,
                "response_format": response_format,
                "started_at_utc": started_at,
                "status": "ok",
                "model": response.model or self.model,
                "usage": usage,
            }
        )
        return response


def run_live_pilot(
    *,
    prompts_path: str | Path = DEFAULT_PROMPTS_PATH,
    reviews_path: str | Path = DEFAULT_REVIEWS_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    run_id: str | None = None,
    dataset_name: str = "amazon_all_beauty_programmatic_gold_live_pilot",
    product_name: str = "amazon all beauty",
    provider_name: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
    semantic_backend: str = "faiss",
    max_prompts: int = 15,
    max_api_failures: int = 3,
    evidence_id: str = DEFAULT_EVIDENCE_ID,
    prompt_ids: Sequence[str] | None = None,
) -> BenchmarkArtifacts:
    prompts_path = Path(prompts_path)
    reviews_path = Path(reviews_path)
    output_root = Path(output_dir)
    run_id = run_id or f"live_gpt4o_pilot_{datetime.now(timezone.utc).strftime('%Y%m%d')}_{uuid4().hex[:4]}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    dotenv_values = read_dotenv(PROJECT_ROOT / ".env")
    live_env = dict(dotenv_values)
    live_env.update(
        {
            "LLM_REVIEW_PROJECT_ROOT": str(run_dir),
            "REVIEWS_DB_PATH": str(run_dir / "runtime" / "proposed_multi_agent.db"),
            "OUTPUT_DIR": str(run_dir / "charts" / "proposed_multi_agent"),
            "VECTORSTORE_DIR": str(run_dir / "vectorstores" / "proposed_multi_agent"),
            "LLM_PROVIDER": provider_name,
            "LLM_MODEL": model,
            "SEMANTIC_RETRIEVAL_BACKEND": semantic_backend,
            "ALLOW_LIVE_LLM": "true",
            "ALLOW_LIVE_RETRIEVAL": "false",
        }
    )
    _install_process_environment(live_env)
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not available for the approved live pilot.")

    settings = load_settings(live_env)
    ensure_directories(settings)
    wrapped_provider = build_llm_provider(settings)
    provider = UsageTrackingProvider(
        wrapped_provider,
        provider_name=type(wrapped_provider).__name__,
        model=getattr(wrapped_provider, "model", model),
    )
    mode_spec = BENCHMARK_MODES["proposed_multi_agent"]
    selected_prompt_ids = tuple(prompt_ids or ())
    prompts = [
        prompt
        for prompt in load_gold_benchmark_prompts(prompts_path)
        if prompt.gold_verification_status == "programmatically_verified"
        and (not selected_prompt_ids or prompt.prompt_id in selected_prompt_ids)
    ][:max_prompts]
    review_rows = json.loads(reviews_path.read_text(encoding="utf-8"))
    results: list[BenchmarkResult] = []
    api_failure_count = 0
    stopped_early_reason: str | None = None

    with sqlite3.connect(settings.database_path) as conn:
        conn.row_factory = sqlite3.Row
        RetrievalAgent(settings).load_records(conn, product_name, review_rows)
        orchestrator = ReviewOrchestrator(settings, provider)
        for prompt in prompts:
            provider.start_prompt(prompt.prompt_id)
            result = _run_prompt(run_id, mode_spec, orchestrator, conn, prompt, provider.model)
            usage = provider.finish_prompt()
            result = _with_live_usage(result, provider, usage)
            results.append(result)
            if _prompt_had_api_failure(usage):
                api_failure_count += 1
            if api_failure_count >= max_api_failures:
                stopped_early_reason = f"Stopped after {api_failure_count} provider/API failure(s)."
                break

    metrics = summarize_benchmark_results(results)
    token_summary = _token_summary(results)
    category_counts = dict(sorted(Counter(result.category for result in results).items()))
    output_files: dict[str, str] = {}
    manifest = {
        "run_id": run_id,
        "mode": "live",
        "modes": [mode_spec.name],
        "dataset_name": dataset_name,
        "prompts_path": str(prompts_path),
        "reviews_path": str(reviews_path),
        "output_dir": str(run_dir),
        "prompt_count": len(prompts),
        "executed_prompt_count": len(results),
        "result_count": len(results),
        "live_mode": True,
        "model_provider": provider.provider_name,
        "model": provider.model,
        "temperature": "provider_default",
        "max_tokens": "provider_default",
        "provider_config": _scrubbed_provider_config(live_env),
        "semantic_retrieval_backend": settings.semantic_retrieval_backend,
        "command": " ".join(sys.argv),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "code_version": _git_commit_sha(),
        "evaluation_tags": list(BENCHMARK_EVALUATION_TAGS),
        "gold_schema_required": True,
        "gold_inclusion_rule": "programmatically_verified prompts only; author/human-unverified judgment prompts excluded",
        "selected_prompt_ids": list(selected_prompt_ids),
        "cost_control": {
            "pre_run_estimate_usd": "< 0.50 rough pilot-control estimate; exact pricing not configured in repository",
            "max_prompts": max_prompts,
            "max_api_failures": max_api_failures,
            "estimated_cost_usd": None,
            "cost_estimation_note": "Token usage is recorded when available; dollar cost remains null unless reliable model pricing is configured.",
        },
        "stopped_early_reason": stopped_early_reason,
    }
    summary = {
        "run_id": run_id,
        "mode": "live",
        "dataset_name": dataset_name,
        "modes": [mode_spec.name],
        "prompt_categories": category_counts,
        "evaluation_tags": list(BENCHMARK_EVALUATION_TAGS),
        "gold_schema_required": True,
        "metrics": metrics,
        "token_usage": token_summary,
        "latency_ms": latency_summary(result.latency_ms for result in results),
        "pilot_status": "bounded live pilot; not final full benchmark",
        "limitations": _pilot_limitations(),
        "output_files": output_files,
        "stopped_early_reason": stopped_early_reason,
    }
    evidence = {
        "evidence_id": evidence_id,
        "run_id": run_id,
        "date_time_utc": datetime.now(timezone.utc).isoformat(),
        "live_mock_status": "live GPT-4o pilot; proposed multi-agent path only",
        "model_provider": provider.provider_name,
        "model": provider.model,
        "input_data": {
            "prompts_path": str(prompts_path),
            "reviews_path": str(reviews_path),
            "dataset_name": dataset_name,
            "product_name": product_name,
        },
        "prompt_count": len(prompts),
        "executed_prompt_count": len(results),
        "prompt_categories": category_counts,
        "output_files": output_files,
        "key_results": metrics,
        "token_usage": token_summary,
        "evaluation_tags": list(BENCHMARK_EVALUATION_TAGS),
        "claim_boundary": (
            "Pilot live evidence for the proposed GPT-4o/LangChain multi-agent path. "
            "Not a final full benchmark, not a user study, and not human-annotated ground truth."
        ),
        "limitations": _pilot_limitations(),
    }
    cost_latency = {
        "run_id": run_id,
        "live_mode": True,
        "model_provider": provider.provider_name,
        "model": provider.model,
        "token_usage": token_summary,
        "estimated_cost_usd": None,
        "cost_estimation_note": "Exact dollar cost unavailable because repository does not configure authoritative current model prices.",
        "latency_ms": latency_summary(result.latency_ms for result in results),
    }

    manifest_path = run_dir / "manifest.json"
    results_path = run_dir / "results.jsonl"
    summary_path = run_dir / "summary.json"
    evidence_path = run_dir / "evidence.json"
    cost_latency_path = run_dir / "cost_latency.json"
    output_files.update(
        {
            "manifest": str(manifest_path),
            "results": str(results_path),
            "summary": str(summary_path),
            "evidence": str(evidence_path),
            "cost_latency": str(cost_latency_path),
        }
    )
    _write_json(manifest_path, manifest)
    _write_jsonl(results_path, [result.to_dict() for result in results])
    _write_json(summary_path, summary)
    _write_json(evidence_path, evidence)
    _write_json(cost_latency_path, cost_latency)
    return BenchmarkArtifacts(
        run_id=run_id,
        run_dir=run_dir,
        manifest_path=manifest_path,
        results_path=results_path,
        summary_path=summary_path,
        evidence_path=evidence_path,
    )


def read_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def _install_process_environment(values: Mapping[str, str]) -> None:
    for key, value in values.items():
        if value:
            os.environ[key] = str(value)


def _with_live_usage(result: BenchmarkResult, provider: UsageTrackingProvider, usage: dict[str, Any]) -> BenchmarkResult:
    return replace(
        result,
        model_provider=provider.provider_name,
        model=provider.model,
        input_tokens=_optional_int(usage.get("input_tokens")),
        output_tokens=_optional_int(usage.get("output_tokens")),
        total_tokens=_optional_int(usage.get("total_tokens")),
        estimated_cost_usd=None,
        token_usage=usage,
        mode_execution_type="live_gpt4o",
        uses_live_gpt4o=True,
        uses_mock_provider=False,
        live_call_count=_optional_int(usage.get("call_count")) or 0,
        mock_call_count=0,
    )


def _aggregate_call_usage(calls: list[dict[str, Any]]) -> dict[str, Any]:
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    token_values_available = False
    for call in calls:
        usage = call.get("usage") if isinstance(call, dict) else {}
        if not isinstance(usage, dict):
            continue
        input_value = _extract_token_value(usage, ("input_tokens", "prompt_tokens"))
        output_value = _extract_token_value(usage, ("output_tokens", "completion_tokens"))
        total_value = _extract_token_value(usage, ("total_tokens",))
        if input_value is not None:
            input_tokens += input_value
            token_values_available = True
        if output_value is not None:
            output_tokens += output_value
            token_values_available = True
        if total_value is not None:
            total_tokens += total_value
            token_values_available = True
    if token_values_available and total_tokens == 0:
        total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens if token_values_available else None,
        "output_tokens": output_tokens if token_values_available else None,
        "total_tokens": total_tokens if token_values_available else None,
        "usage_available": token_values_available,
        "call_count": len(calls),
        "api_error_count": sum(1 for call in calls if call.get("status") == "error"),
    }


def _extract_token_value(usage: Mapping[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return None


def _optional_int(value: Any) -> int | None:
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _prompt_had_api_failure(usage: dict[str, Any]) -> bool:
    return int(usage.get("api_error_count") or 0) > 0


def _token_summary(results: list[BenchmarkResult]) -> dict[str, Any]:
    usage_available = any(result.total_tokens is not None for result in results)
    return {
        "usage_available": usage_available,
        "input_tokens": _sum_optional(result.input_tokens for result in results),
        "output_tokens": _sum_optional(result.output_tokens for result in results),
        "total_tokens": _sum_optional(result.total_tokens for result in results),
        "call_count": sum(int((result.token_usage or {}).get("call_count") or 0) for result in results),
        "api_error_count": sum(int((result.token_usage or {}).get("api_error_count") or 0) for result in results),
        "estimated_cost_usd": None,
        "cost_estimation_note": "Dollar-cost calculation is intentionally left null without configured authoritative current model pricing.",
    }


def _sum_optional(values: Any) -> int | None:
    filtered = [int(value) for value in values if value is not None]
    if not filtered:
        return None
    return sum(filtered)


def _scrubbed_provider_config(values: Mapping[str, str]) -> dict[str, Any]:
    return {
        key: ("<present redacted>" if _looks_like_secret_key(key) and bool(value) else value)
        for key, value in sorted(values.items())
        if key not in {"PATH", "Path"}
    }


def _looks_like_secret_key(key: str) -> bool:
    upper = key.upper()
    return key in SECRET_KEY_NAMES or any(fragment in upper for fragment in ("KEY", "SECRET", "TOKEN", "PASSWORD"))


def _git_commit_sha() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    sha = completed.stdout.strip()
    return sha or None


def _pilot_limitations() -> list[str]:
    return [
        "Small bounded live pilot only; not a final full benchmark.",
        "Gold items are programmatically verified from local data, not human/adjudicated annotations.",
        "Mock results remain infrastructure/control evidence only and are not proposed-system performance evidence.",
        "No user study, no inter-annotator agreement, and no human translation-quality evaluation are claimed.",
        "Alternative/open-source model comparison remains deferred to the final/optional stage.",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a bounded live GPT-4o pilot over programmatically verified gold prompts.")
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS_PATH)
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dataset-name", default="amazon_all_beauty_programmatic_gold_live_pilot")
    parser.add_argument("--product-name", default="amazon all beauty")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER, choices=("langchain", "openai"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--semantic-backend", default="faiss", choices=("faiss", "lexical"))
    parser.add_argument("--max-prompts", type=int, default=15)
    parser.add_argument("--max-api-failures", type=int, default=3)
    parser.add_argument("--evidence-id", default=DEFAULT_EVIDENCE_ID)
    parser.add_argument("--prompt-ids", nargs="*", default=None, help="Optional prompt IDs for a tiny targeted validation run.")
    args = parser.parse_args()

    artifacts = run_live_pilot(
        prompts_path=args.prompts,
        reviews_path=args.reviews,
        output_dir=args.output_dir,
        run_id=args.run_id,
        dataset_name=args.dataset_name,
        product_name=args.product_name,
        provider_name=args.provider,
        model=args.model,
        semantic_backend=args.semantic_backend,
        max_prompts=args.max_prompts,
        max_api_failures=args.max_api_failures,
        evidence_id=args.evidence_id,
        prompt_ids=args.prompt_ids,
    )
    print(f"run_id={artifacts.run_id}")
    print(f"manifest={artifacts.manifest_path}")
    print(f"results={artifacts.results_path}")
    print(f"summary={artifacts.summary_path}")
    print(f"evidence={artifacts.evidence_path}")


if __name__ == "__main__":
    main()

