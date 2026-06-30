from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import sys
import time
from typing import Any, Mapping, Sequence
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (SRC_ROOT, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from evaluation.live_pilot import (  # noqa: E402
    UsageTrackingProvider,
    _aggregate_call_usage,
    _git_commit_sha,
    _install_process_environment,
    read_dotenv,
)
from llm_review_analysis.agents.semantic_reasoning_agent import SemanticReasoningAgent  # noqa: E402
from llm_review_analysis.config import ensure_directories, load_settings  # noqa: E402
from llm_review_analysis.db.schema import ensure_review_table, insert_review_rows  # noqa: E402
from llm_review_analysis.providers import build_llm_provider  # noqa: E402


DEFAULT_ITEMS_PATH = PROJECT_ROOT / "evaluation" / "live_controlled_component_ablation_items.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "benchmarks"
DEFAULT_PROVIDER = "langchain"
DEFAULT_MODEL = "gpt-4o"
DEFAULT_SEMANTIC_BACKEND = "faiss"
COMPONENT_ORDER = (
    "orchestrator_routing",
    "language_translation",
    "topic_assignment",
    "semantic_tags",
    "vector_retrieval",
    "data_analytics",
)
COMPONENT_LABELS = {
    "orchestrator_routing": "Orchestrator / routing",
    "language_translation": "Language and Translation Agent",
    "topic_assignment": "Topic Assignment Agent",
    "semantic_tags": "Semantics Agent semantic tags",
    "vector_retrieval": "Vector retrieval in Semantics route",
    "data_analytics": "Data Analytics Agent",
}
ROUTE_LABELS = {
    "DIRECT_SQL",
    "SEMANTICS",
    "ANALYTICS",
    "TRANSLATION_MULTILINGUAL",
    "UNSUPPORTED_CLARIFICATION",
}
COMPONENT_ALIASES = {
    "routing": "orchestrator_routing",
    "orchestrator": "orchestrator_routing",
    "orchestrator_routing": "orchestrator_routing",
    "language": "language_translation",
    "translation": "language_translation",
    "language_translation": "language_translation",
    "topic": "topic_assignment",
    "topic_assignment": "topic_assignment",
    "tags": "semantic_tags",
    "semantic_tags": "semantic_tags",
    "vector": "vector_retrieval",
    "vector_retrieval": "vector_retrieval",
    "analytics": "data_analytics",
    "data_analytics": "data_analytics",
}
LANGUAGE_ALIASES = {
    "arabic": {"arabic", "ar"},
    "chinese": {"chinese", "zh", "zh-cn", "zh-hans", "mandarin"},
    "french": {"french", "fr"},
    "spanish": {"spanish", "es"},
    "german": {"german", "de"},
    "hindi": {"hindi", "hi"},
    "portuguese": {"portuguese", "pt"},
    "italian": {"italian", "it"},
}
TRANSLATION_SYNONYMS = {
    "lunch": ("lunch", "noon"),
    "support": ("support", "customer support", "customer service"),
    "replied": ("replied", "responded"),
    "great": ("great", "excellent"),
    "two days": ("two days", "two whole days"),
    "older tablet": ("older tablet", "old tablet"),
    "same review": ("same review", "same evaluation"),
    "detail": ("detail", "details"),
    "drains": ("drains", "drain", "runs out"),
}
ALLOWED_CHART_TYPES = {"bar", "line", "pie", "unsupported"}
ALLOWED_GROUP_FIELDS = {"rating", "country", "date", "topic", "verified"}
ALLOWED_AGGREGATIONS = {"count", "avg", "sum", "unsupported", ""}
REQUIRED_OUTPUT_FILES = (
    "items_used.json",
    "results.jsonl",
    "summary.csv",
    "summary.json",
    "summary.md",
    "table_ready_live_controlled_component_ablation.csv",
    "failure_cases.md",
    "judge_outputs.jsonl",
)


@dataclass(frozen=True)
class LiveCallResult:
    parsed: dict[str, Any]
    raw: str
    usage: dict[str, Any]
    latency_ms: int
    error: str | None = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a live controlled component-level ablation over fixed local inputs.")
    parser.add_argument("--items", default=str(DEFAULT_ITEMS_PATH), help="Path to live controlled item JSON.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Benchmark output root.")
    parser.add_argument("--preflight", action="store_true", help="Validate design and print live-run readiness without API calls.")
    parser.add_argument("--live", action="store_true", help="Run the live controlled benchmark once.")
    parser.add_argument("--allow-live", action="store_true", help="Required explicit gate for live model and embedding calls.")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER, help="LLM provider for live calls.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model id for live calls.")
    parser.add_argument("--semantic-backend", default=DEFAULT_SEMANTIC_BACKEND, help="Semantic retrieval backend for vector component.")
    parser.add_argument("--embedding-model", default="text-embedding-3-small", help="Embedding model used by FAISS semantic retrieval.")
    parser.add_argument("--run-id", default=None, help="Optional output run id.")
    parser.add_argument("--max-api-errors", type=int, default=8, help="Stop after this many live API failures.")
    parser.add_argument(
        "--components",
        default="all",
        help="Comma-separated component filter. Supports all, routing, translation, topic, tags, vector, analytics.",
    )
    parser.add_argument(
        "--merge-from",
        default=None,
        help="Optional previous live controlled output folder. Components not rerun are preserved in merged outputs.",
    )
    args = parser.parse_args()

    data = load_item_data(Path(args.items))
    if args.preflight:
        report = build_preflight_report(args, data)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if not report["preflight_passed"]:
            raise SystemExit(1)
        return
    if args.live:
        if not args.allow_live:
            raise SystemExit("--live requires --allow-live after author approval.")
        report = build_preflight_report(args, data)
        if not report["preflight_passed"]:
            print(json.dumps(report, indent=2, ensure_ascii=False))
            raise SystemExit(1)
        if not report["required_api_keys"]["OPENAI_API_KEY"]["available"]:
            raise SystemExit("OPENAI_API_KEY is not available in the environment or .env; live run not started.")
        run_dir = run_live_benchmark(args, data, report)
        print(str(run_dir))
        return
    parser.print_help()


def load_item_data(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw.get("review_rows")
    items = raw.get("items")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Item file must contain non-empty review_rows.")
    if not isinstance(items, list) or not items:
        raise ValueError("Item file must contain non-empty items.")
    with_ids = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise ValueError(f"Review row {index} is not an object.")
        merged = {"id": index}
        merged.update(row)
        with_ids.append(merged)
    raw["review_rows_with_ids"] = with_ids
    validate_items(raw)
    return raw


def validate_items(data: Mapping[str, Any]) -> None:
    counts = Counter(str(item.get("component", "")) for item in data["items"])
    missing = [component for component in COMPONENT_ORDER if counts[component] == 0]
    if missing:
        raise ValueError(f"Missing component item groups: {', '.join(missing)}")
    low_counts = [component for component in COMPONENT_ORDER if counts[component] < 15]
    if low_counts:
        raise ValueError(f"Components below 15 items: {', '.join(low_counts)}")
    for item in data["items"]:
        component = str(item.get("component", ""))
        if component not in COMPONENT_ORDER:
            raise ValueError(f"Unsupported component for {item.get('id')}: {component}")
        if component == "orchestrator_routing" and item.get("expected_route") not in ROUTE_LABELS:
            raise ValueError(f"Unsupported expected route for {item.get('id')}: {item.get('expected_route')}")
        if component in {"topic_assignment", "semantic_tags", "vector_retrieval"} and not item.get("expected_ids"):
            raise ValueError(f"{item.get('id')} must define expected_ids.")


def parse_component_filter(raw: str | None) -> tuple[str, ...]:
    if raw is None or raw.strip().lower() == "all":
        return COMPONENT_ORDER
    selected: list[str] = []
    for value in raw.split(","):
        key = value.strip().lower().replace("-", "_")
        if not key:
            continue
        component = COMPONENT_ALIASES.get(key)
        if component is None:
            raise ValueError(f"Unknown component filter value: {value}")
        if component not in selected:
            selected.append(component)
    if not selected:
        raise ValueError("At least one component must be selected.")
    return tuple(selected)


def filter_items(data: Mapping[str, Any], selected_components: Sequence[str]) -> list[Mapping[str, Any]]:
    selected = set(selected_components)
    return [item for item in data["items"] if str(item.get("component")) in selected]


def build_preflight_report(args: argparse.Namespace, data: Mapping[str, Any]) -> dict[str, Any]:
    dotenv_values = read_dotenv(PROJECT_ROOT / ".env")
    merged_env = dict(dotenv_values)
    merged_env.update(os.environ)
    selected_components = parse_component_filter(args.components)
    counts = item_counts(data, selected_components)
    dependency_status = dependency_checks(args.provider, args.semantic_backend)
    expected_calls = expected_workflow_calls(counts, selected_components)
    embedding_batches = counts.get("vector_retrieval", 0) if args.semantic_backend.lower() == "faiss" else 0
    count_passed = all(counts.get(component, 0) >= 20 for component in selected_components)
    dependency_passed = all(item["available"] for item in dependency_status.values())
    return {
        "preflight_passed": bool(count_passed and dependency_passed),
        "selected_components": [COMPONENT_LABELS[k] for k in selected_components],
        "merge_from": args.merge_from,
        "item_counts_per_component": {COMPONENT_LABELS[k]: counts.get(k, 0) for k in selected_components},
        "data_artifacts_used": {
            "items_path": str(Path(args.items).resolve()),
            "local_review_row_count": len(data["review_rows"]),
            "product_table": data.get("product_table"),
            "external_retrieval": False,
            "manuscript_insertion": False,
        },
        "full_conditions": full_conditions(),
        "restricted_conditions": restricted_conditions(),
        "masking_or_disabling_mechanism": masking_mechanisms(),
        "expected_workflow_model_calls": expected_calls,
        "expected_judge_calls": 0,
        "expected_embedding_batches": embedding_batches,
        "required_api_keys": {
            "OPENAI_API_KEY": {
                "required_for": ["langchain GPT-4o calls", "OpenAI embeddings for FAISS"],
                "available": bool(merged_env.get("OPENAI_API_KEY")),
            }
        },
        "dependency_status": dependency_status,
        "estimated_cost": {
            "pricing_configured": False,
            "estimated_cost_usd": None,
            "note": "No authoritative pricing is configured in this runner; token counts are recorded after live execution.",
        },
        "exact_live_command": exact_live_command(args),
        "risks_and_limitations": [
            "Controlled local fixtures do not demonstrate population-level generalizability.",
            "Live LLM outputs may vary across model versions and repeated executions.",
            "No LLM judge is used; text outputs are scored through deterministic JSON fields and expected IDs/labels.",
            "The FAISS path uses live embeddings over a fixed local corpus; embedding token usage may not be reported by the chat provider.",
            "Restricted conditions are diagnostic masks/fallbacks and are not alternative production systems.",
        ],
        "required_output_files": list(REQUIRED_OUTPUT_FILES),
        "code_version": _git_commit_sha(),
    }


def item_counts(data: Mapping[str, Any], selected_components: Sequence[str] | None = None) -> dict[str, int]:
    selected = set(selected_components or COMPONENT_ORDER)
    return dict(Counter(str(item.get("component", "")) for item in data["items"] if str(item.get("component", "")) in selected))


def dependency_checks(provider: str, semantic_backend: str) -> dict[str, dict[str, Any]]:
    checks: dict[str, dict[str, Any]] = {}
    if provider.lower() in {"langchain", "langchain-openai", "langchain_openai"}:
        checks["langchain_openai"] = _module_status("langchain_openai")
    if semantic_backend.lower() == "faiss":
        checks["langchain_community.vectorstores"] = _module_status("langchain_community.vectorstores")
        checks["langchain_text_splitters_or_langchain"] = {
            "available": bool(_find_spec("langchain_text_splitters") or _find_spec("langchain")),
            "note": "Required for RecursiveCharacterTextSplitter.",
        }
    return checks


def _module_status(name: str) -> dict[str, Any]:
    return {"available": _find_spec(name) is not None}


def _find_spec(name: str) -> Any:
    try:
        return importlib.util.find_spec(name)
    except (ImportError, AttributeError, ValueError):
        return None


def expected_workflow_calls(counts: Mapping[str, int], selected_components: Sequence[str]) -> dict[str, Any]:
    by_component = {
        "orchestrator_routing": counts.get("orchestrator_routing", 0),
        "language_translation": counts.get("language_translation", 0),
        "topic_assignment": counts.get("topic_assignment", 0) * 2,
        "semantic_tags": counts.get("semantic_tags", 0) * 2,
        "vector_retrieval": counts.get("vector_retrieval", 0) * 2,
        "data_analytics": counts.get("data_analytics", 0),
    }
    by_component = {component: value for component, value in by_component.items() if component in selected_components}
    return {
        "by_component": {COMPONENT_LABELS[k]: v for k, v in by_component.items()},
        "total": sum(by_component.values()),
    }


def full_conditions() -> dict[str, str]:
    return {
        "Orchestrator / routing": "Live LLM route-selection prompt over fixed prompts and controlled route labels.",
        "Language and Translation Agent": "Live LLM detects language, translates the snippet, preserves intent, and answers in the original language.",
        "Topic Assignment Agent": "Live LLM receives fixed local review rows with topic labels available in context.",
        "Semantics Agent semantic tags": "Live LLM receives fixed local review rows with semantic tags available in context.",
        "Vector retrieval in Semantics route": "Project FAISS semantic retrieval over the fixed local corpus followed by a live LLM reasoning answer.",
        "Data Analytics Agent": "Live LLM creates a constrained chart specification that is validated against approved fields and aggregations.",
    }


def restricted_conditions() -> dict[str, str]:
    return {
        "Orchestrator / routing": "Simplified rule-based/default routing.",
        "Language and Translation Agent": "Translation hidden/disabled; non-English tasks return controlled unsupported behavior.",
        "Topic Assignment Agent": "Topic labels masked from the live LLM context.",
        "Semantics Agent semantic tags": "Semantic tags masked from the live LLM context.",
        "Vector retrieval in Semantics route": "FAISS retrieval disabled and replaced with a static first-k evidence baseline.",
        "Data Analytics Agent": "Analytics disabled; chart requests receive direct-SQL-only controlled fallback responses.",
    }


def masking_mechanisms() -> dict[str, str]:
    return {
        "topic_assignment": "The restricted prompt omits the topic column and only shows id, title, rating, and review text.",
        "semantic_tags": "The restricted prompt omits the semantic_tags column and only shows id, title, rating, and review text.",
        "language_translation": "The restricted condition does not translate or expose translated text.",
        "vector_retrieval": "The restricted condition uses review IDs 1, 2, and 3 as static first-k evidence for every query.",
        "data_analytics": "The restricted condition does not request or validate chart specifications.",
        "orchestrator_routing": "The restricted condition uses keyword rules without live model routing.",
    }


def exact_live_command(args: argparse.Namespace) -> str:
    command = (
        "python evaluation/run_live_controlled_component_ablation.py "
        f"--live --allow-live --provider {args.provider} --model {args.model} "
        f"--semantic-backend {args.semantic_backend}"
    )
    if args.components and args.components != "all":
        command += f" --components {args.components}"
    if args.merge_from:
        command += f" --merge-from {args.merge_from}"
    return command


def run_live_benchmark(args: argparse.Namespace, data: Mapping[str, Any], preflight: Mapping[str, Any]) -> Path:
    run_id = args.run_id or f"live_controlled_component_ablation_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:6]}"
    run_root = Path(args.output_dir)
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "items_used.json", data)
    write_json(run_dir / "preflight.json", preflight)
    selected_components = parse_component_filter(args.components)
    selected_items = filter_items(data, selected_components)

    dotenv_values = read_dotenv(PROJECT_ROOT / ".env")
    live_env = dict(dotenv_values)
    live_env.update(os.environ)
    live_env.update(
        {
            "LLM_REVIEW_PROJECT_ROOT": str(run_dir),
            "REVIEWS_DB_PATH": str(run_dir / "runtime" / "live_controlled.db"),
            "OUTPUT_DIR": str(run_dir / "charts"),
            "VECTORSTORE_DIR": str(run_dir / "vectorstores"),
            "LLM_PROVIDER": args.provider,
            "LLM_MODEL": args.model,
            "LLM_TEMPERATURE": "0",
            "LLM_MAX_TOKENS": "700",
            "SEMANTIC_RETRIEVAL_BACKEND": args.semantic_backend,
            "EMBEDDING_MODEL": args.embedding_model,
            "ALLOW_LIVE_LLM": "true",
            "ALLOW_LIVE_RETRIEVAL": "false",
        }
    )
    _install_process_environment(live_env)
    settings = load_settings(live_env)
    ensure_directories(settings)
    wrapped_provider = build_llm_provider(settings)
    provider = UsageTrackingProvider(
        wrapped_provider,
        provider_name=type(wrapped_provider).__name__,
        model=getattr(wrapped_provider, "model", args.model),
    )

    api_error_count = 0
    results: list[dict[str, Any]] = []
    with sqlite3.connect(settings.database_path) as conn:
        conn.row_factory = sqlite3.Row
        table = str(data["product_table"])
        ensure_review_table(conn, table)
        insert_review_rows(conn, table, data["review_rows"])
        semantic_agent = SemanticReasoningAgent(settings=settings, provider=provider, backend=args.semantic_backend, top_k=3)
        rows = list(data["review_rows_with_ids"])
        for item in selected_items:
            component = str(item["component"])
            component_results = run_item(component, item, rows, provider, semantic_agent, conn, table)
            results.extend(component_results)
            api_error_count += sum(1 for row in component_results if row.get("api_error"))
            append_jsonl(run_dir / "results.jsonl", component_results)
            if api_error_count >= args.max_api_errors:
                results.append(
                    {
                        "component": "run_control",
                        "item_id": "stopped_early",
                        "condition": "control",
                        "success": False,
                        "quality_score": 0.0,
                        "failure_mode": "max_api_errors_reached",
                        "score_type": "deterministic",
                        "notes": f"Stopped after {api_error_count} API/model error rows.",
                    }
                )
                break

    write_judge_outputs(run_dir / "judge_outputs.jsonl")
    summary = summarize_results(results, run_id=run_id, data=data, args=args, preflight=preflight, components=selected_components)
    write_json(run_dir / "summary.json", summary)
    write_summary_csv(run_dir / "summary.csv", summary["component_summaries"])
    write_table_ready_csv(run_dir / "table_ready_live_controlled_component_ablation.csv", summary["table_ready_rows"])
    write_summary_md(run_dir / "summary.md", summary)
    write_failure_cases(run_dir / "failure_cases.md", results)
    if args.merge_from:
        merge_dir = Path(args.merge_from)
        preserved_results = load_jsonl(merge_dir / "results.jsonl")
        selected_set = set(selected_components)
        merged_results = [row for row in preserved_results if row.get("component") not in selected_set]
        merged_results.extend(row for row in results if row.get("condition") in {"full", "restricted"})
        write_jsonl(run_dir / "merged_results.jsonl", merged_results)
        merged_summary = summarize_results(
            merged_results,
            run_id=run_id + "_merged",
            data=data,
            args=args,
            preflight=preflight,
            components=COMPONENT_ORDER,
        )
        merged_summary["merge"] = {
            "merge_from": str(merge_dir),
            "rerun_components": list(selected_components),
            "preserved_components": [component for component in COMPONENT_ORDER if component not in selected_set],
        }
        write_json(run_dir / "merged_summary.json", merged_summary)
        write_summary_csv(run_dir / "merged_summary.csv", merged_summary["component_summaries"])
        write_table_ready_csv(run_dir / "merged_table_ready_live_controlled_component_ablation.csv", merged_summary["table_ready_rows"])
        write_summary_md(run_dir / "merged_summary.md", merged_summary)
        write_failure_cases(run_dir / "merged_failure_cases.md", merged_results)
    return run_dir


def run_item(
    component: str,
    item: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    provider: UsageTrackingProvider,
    semantic_agent: SemanticReasoningAgent,
    conn: sqlite3.Connection,
    table: str,
) -> list[dict[str, Any]]:
    if component == "orchestrator_routing":
        return run_route_item(item, provider)
    if component == "language_translation":
        return run_language_item(item, provider)
    if component == "topic_assignment":
        return run_id_selection_item(item, rows, provider, component, include_topic=True, include_tags=False)
    if component == "semantic_tags":
        return run_id_selection_item(item, rows, provider, component, include_topic=False, include_tags=True)
    if component == "vector_retrieval":
        return run_vector_item(item, rows, provider, semantic_agent, conn, table)
    if component == "data_analytics":
        return run_analytics_item(item, rows, provider)
    raise ValueError(f"Unsupported component: {component}")


def run_route_item(item: Mapping[str, Any], provider: UsageTrackingProvider) -> list[dict[str, Any]]:
    prompt = (
        "Classify the user prompt for a multilingual product-review analysis orchestrator. "
        "Return JSON only with fields route and reason. "
        "Allowed route values: DIRECT_SQL, SEMANTICS, ANALYTICS, TRANSLATION_MULTILINGUAL, UNSUPPORTED_CLARIFICATION. "
        "Use DIRECT_SQL for exact scalar facts that can be answered from database fields. "
        "Use SEMANTICS for interpretive review-evidence questions. "
        "Use ANALYTICS for chart, trend, grouping, or visualization requests. "
        "If the prompt is non-English, classify the underlying task after translation; use TRANSLATION_MULTILINGUAL only when the user's primary request is translation or language handling itself. "
        "Use UNSUPPORTED_CLARIFICATION when the prompt depends on missing prior context or lacks a clear product-review task.\n"
        f"User prompt: {item['prompt']}"
    )
    full = call_json(provider, f"{item['id']}:full", prompt, purpose="live_route_selection")
    full_route = normalize_route(full.parsed.get("route"))
    full_score = 1.0 if full_route == item["expected_route"] else 0.0
    restricted_route = restricted_route_decision(str(item["prompt"]))
    restricted_score = 1.0 if restricted_route == item["expected_route"] else 0.0
    return [
        result_row(item, "full", full_score == 1.0, full_score, "none" if full_score == 1.0 else "routing_error", full, actual_route=full_route),
        result_row(
            item,
            "restricted",
            restricted_score == 1.0,
            restricted_score,
            "none" if restricted_score == 1.0 else "routing_error",
            static_output={"route": restricted_route, "reason": "keyword/default restricted router"},
            actual_route=restricted_route,
        ),
    ]


def run_language_item(item: Mapping[str, Any], provider: UsageTrackingProvider) -> list[dict[str, Any]]:
    prompt = (
        "You are the live Language and Translation Agent for a product-review workflow. "
        "Detect the review language, translate the review into English, answer the English question using the review, "
        "and keep the final answer language consistent with the review language. Return JSON only with fields "
        "detected_language, english_translation, answer_label, response_language, answer. "
        "answer_label must be one of yes, no, unclear.\n"
        f"Review snippet: {item['review_snippet']}\n"
        f"Question: {item['question']}"
    )
    full = call_json(provider, f"{item['id']}:full", prompt, purpose="live_language_translation")
    full_score, details = score_language(full.parsed, item)
    restricted_output = {
        "detected_language": item.get("language"),
        "answer_label": "unsupported",
        "response_language": "English",
        "answer": "Translation is disabled in this restricted condition, so the non-English review cannot be analyzed.",
        "controlled_unsupported": True,
    }
    restricted_score = 0.25
    return [
        result_row(
            item,
            "full",
            full_score >= 0.75,
            full_score,
            "none" if full_score >= 0.75 else "translation_or_language_error",
            full,
            score_details=details,
        ),
        result_row(
            item,
            "restricted",
            False,
            restricted_score,
            "translation_disabled",
            static_output=restricted_output,
            score_details={"controlled_unsupported": True},
        ),
    ]


def run_id_selection_item(
    item: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    provider: UsageTrackingProvider,
    component: str,
    *,
    include_topic: bool,
    include_tags: bool,
) -> list[dict[str, Any]]:
    full_context = format_review_context(rows, include_topic=include_topic, include_tags=include_tags)
    restricted_context = format_review_context(rows, include_topic=False, include_tags=False)
    component_name = COMPONENT_LABELS[component]
    full_prompt = id_selection_prompt(component_name, item["prompt"], full_context)
    restricted_prompt = id_selection_prompt(component_name + " restricted", item["prompt"], restricted_context)
    full = call_json(provider, f"{item['id']}:full", full_prompt, purpose=f"live_{component}_full")
    restricted = call_json(provider, f"{item['id']}:restricted", restricted_prompt, purpose=f"live_{component}_restricted")
    expected = set(int(value) for value in item["expected_ids"])
    full_ids = parse_ids(full.parsed)
    restricted_ids = parse_ids(restricted.parsed)
    full_score = f1_score(expected, full_ids)
    restricted_score = f1_score(expected, restricted_ids)
    return [
        result_row(
            item,
            "full",
            full_score == 1.0,
            full_score,
            "none" if full_score == 1.0 else "id_selection_error",
            full,
            expected_ids=sorted(expected),
            actual_ids=sorted(full_ids),
        ),
        result_row(
            item,
            "restricted",
            restricted_score == 1.0,
            restricted_score,
            "none" if restricted_score == 1.0 else masked_failure_mode(component),
            restricted,
            expected_ids=sorted(expected),
            actual_ids=sorted(restricted_ids),
        ),
    ]


def run_vector_item(
    item: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    provider: UsageTrackingProvider,
    semantic_agent: SemanticReasoningAgent,
    conn: sqlite3.Connection,
    table: str,
) -> list[dict[str, Any]]:
    query = str(item["query"])
    provider.start_prompt(f"{item['id']}:full")
    started = time.perf_counter()
    full_error: str | None = None
    try:
        trace = semantic_agent.answer_with_trace(conn, table, query)
        full_output = {
            "answer": trace.answer,
            "evidence_ids": [safe_int(value) for value in trace.evidence_ids if safe_int(value) is not None],
            "evidence_snippets": list(trace.evidence_snippets),
        }
    except Exception as exc:  # noqa: BLE001 - benchmark must record live failures.
        full_error = f"{type(exc).__name__}: {exc}"
        full_output = {"answer": "", "evidence_ids": [], "evidence_snippets": []}
    full_usage = provider.finish_prompt()
    full_call = LiveCallResult(
        parsed=full_output,
        raw=json.dumps(full_output, ensure_ascii=False),
        usage=full_usage,
        latency_ms=int((time.perf_counter() - started) * 1000),
        error=full_error,
    )

    static_evidence = list(rows[:3])
    restricted_prompt = (
        "Answer the fixed semantic review query using only the restricted static first-k evidence below. "
        "Return JSON only with fields answer and evidence_ids. Do not invent evidence outside the provided rows.\n"
        f"Query: {query}\n"
        f"Evidence:\n{format_review_context(static_evidence, include_topic=False, include_tags=True)}"
    )
    restricted = call_json(provider, f"{item['id']}:restricted", restricted_prompt, purpose="live_vector_static_first_k")
    if not restricted.parsed.get("evidence_ids"):
        restricted.parsed["evidence_ids"] = [1, 2, 3]
    expected = set(int(value) for value in item["expected_ids"])
    full_score = score_vector_output(expected, item.get("expected_keywords", []), full_output)
    restricted_score = score_vector_output(expected, item.get("expected_keywords", []), restricted.parsed)
    return [
        result_row(
            item,
            "full",
            full_score >= 0.75 and not full_error,
            full_score if not full_error else 0.0,
            "none" if full_score >= 0.75 and not full_error else "retrieval_or_grounding_error",
            full_call,
            expected_ids=sorted(expected),
            actual_ids=sorted(parse_ids(full_output)),
            embedding_batches=1,
        ),
        result_row(
            item,
            "restricted",
            restricted_score >= 0.75,
            restricted_score,
            "none" if restricted_score >= 0.75 else "static_first_k_missed_evidence",
            restricted,
            expected_ids=sorted(expected),
            actual_ids=sorted(parse_ids(restricted.parsed)),
            embedding_batches=0,
        ),
    ]


def run_analytics_item(item: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], provider: UsageTrackingProvider) -> list[dict[str, Any]]:
    prompt = (
        "You are the live Data Analytics Agent for a product-review workflow. "
        "Convert the user request into a constrained chart specification over fixed review-table fields. "
        "Allowed chart types: bar, line, pie. Unsupported requested chart types must return supported=false and chart_type=unsupported. "
        "Allowed group_by fields: rating, country, date, topic, verified. Allowed aggregations: count, avg, sum. "
        "Return JSON only with fields supported, chart_type, group_by, aggregation, y_field, explanation.\n"
        f"User request: {item['prompt']}"
    )
    full = call_json(provider, f"{item['id']}:full", prompt, purpose="live_analytics_chart_spec")
    full_score, details = score_chart_spec(full.parsed, item, rows)
    expected_unsupported = item.get("expected_chart_type") == "unsupported"
    restricted_output = {
        "supported": False,
        "chart_type": "unsupported",
        "message": "Analytics is disabled in this restricted condition; direct-SQL-only fallback cannot create charts.",
        "controlled_unsupported": True,
    }
    restricted_score = 1.0 if expected_unsupported else 0.25
    return [
        result_row(
            item,
            "full",
            full_score >= 0.75,
            full_score,
            "none" if full_score >= 0.75 else "invalid_or_incorrect_chart_spec",
            full,
            score_details=details,
        ),
        result_row(
            item,
            "restricted",
            expected_unsupported,
            restricted_score,
            "none" if expected_unsupported else "analytics_disabled",
            static_output=restricted_output,
            score_details={"controlled_unsupported": True},
        ),
    ]


def call_json(provider: UsageTrackingProvider, prompt_id: str, prompt: str, *, purpose: str) -> LiveCallResult:
    provider.start_prompt(prompt_id)
    started = time.perf_counter()
    raw = ""
    parsed: dict[str, Any] = {}
    error_text: str | None = None
    try:
        response = provider.generate(prompt, purpose=purpose, response_format="json")
        raw = response.content.strip()
        parsed = parse_json_object(raw)
    except Exception as exc:  # noqa: BLE001 - benchmark must record live failures.
        error_text = f"{type(exc).__name__}: {exc}"
    usage = provider.finish_prompt()
    return LiveCallResult(parsed=parsed, raw=raw, usage=usage, latency_ms=int((time.perf_counter() - started) * 1000), error=error_text)


def parse_json_object(text: str) -> dict[str, Any]:
    if not text:
        return {}
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned.strip(), flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return {}
        value = json.loads(match.group(0))
    return value if isinstance(value, dict) else {}


def result_row(
    item: Mapping[str, Any],
    condition: str,
    success: bool,
    quality_score: float,
    failure_mode: str,
    call: LiveCallResult | None = None,
    *,
    static_output: Mapping[str, Any] | None = None,
    expected_ids: Sequence[int] | None = None,
    actual_ids: Sequence[int] | None = None,
    actual_route: str | None = None,
    score_details: Mapping[str, Any] | None = None,
    embedding_batches: int = 0,
) -> dict[str, Any]:
    usage = call.usage if call else {"input_tokens": None, "output_tokens": None, "total_tokens": None, "call_count": 0, "api_error_count": 0, "calls": []}
    output = call.parsed if call else dict(static_output or {})
    raw = call.raw if call else json.dumps(output, ensure_ascii=False)
    api_error = bool(call and call.error)
    normalized_quality = max(0.0, min(1.0, float(quality_score)))
    return {
        "component": str(item["component"]),
        "component_label": COMPONENT_LABELS.get(str(item["component"]), str(item["component"])),
        "item_id": str(item["id"]),
        "condition": condition,
        "prompt": item.get("prompt") or item.get("question") or item.get("query"),
        "success": bool(success) and not api_error,
        "quality_score": normalized_quality if not api_error else 0.0,
        "failure_mode": "api_error" if api_error else failure_mode,
        "score_type": "deterministic",
        "judge_assisted": False,
        "expected_route": item.get("expected_route"),
        "actual_route": actual_route,
        "expected_ids": list(expected_ids or []),
        "actual_ids": list(actual_ids or []),
        "expected_chart_type": item.get("expected_chart_type"),
        "expected_group_by": item.get("expected_group_by"),
        "expected_aggregation": item.get("expected_aggregation"),
        "output": output,
        "raw_output": raw,
        "score_details": dict(score_details or {}),
        "workflow_calls": int(usage.get("call_count") or 0),
        "judge_calls": 0,
        "embedding_batches": embedding_batches,
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "token_usage": usage,
        "latency_ms": call.latency_ms if call else 0,
        "api_error": api_error,
        "error": call.error if call else None,
    }


def normalize_route(value: Any) -> str:
    text = str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "SQL": "DIRECT_SQL",
        "DIRECT": "DIRECT_SQL",
        "DATA_ANALYTICS": "ANALYTICS",
        "CHART": "ANALYTICS",
        "VISUALIZATION": "ANALYTICS",
        "TRANSLATION": "TRANSLATION_MULTILINGUAL",
        "MULTILINGUAL": "TRANSLATION_MULTILINGUAL",
        "CLARIFICATION": "UNSUPPORTED_CLARIFICATION",
        "UNSUPPORTED": "UNSUPPORTED_CLARIFICATION",
        "REFUSAL": "UNSUPPORTED_CLARIFICATION",
    }
    return aliases.get(text, text)


def restricted_route_decision(prompt: str) -> str:
    lower = prompt.lower()
    if _requires_clarification(lower):
        return "UNSUPPORTED_CLARIFICATION"
    if any(word in lower for word in ("translate", "translation", "language", "detect language")):
        return "TRANSLATION_MULTILINGUAL"
    if _contains_non_ascii(prompt):
        return "TRANSLATION_MULTILINGUAL"
    if any(word in lower for word in ("chart", "plot", "visualize", "visual", "trend", "distribution")):
        return "ANALYTICS"
    if any(phrase in lower for phrase in ("exact sql", "how many", "count", "average", "number of")):
        return "DIRECT_SQL"
    if any(word in lower for word in ("why", "summarize", "complain", "good", "reliable", "worry", "should", "explain", "evidence")):
        return "SEMANTICS"
    return "DIRECT_SQL"


def _contains_non_ascii(text: str) -> bool:
    return any(ord(char) > 127 for char in text)


def _requires_clarification(lower_prompt: str) -> bool:
    return any(
        phrase in lower_prompt
        for phrase in (
            "tell me about it",
            "previous product",
            "previous conversation",
            "before",
            "that one",
        )
    )


def score_language(parsed: Mapping[str, Any], item: Mapping[str, Any]) -> tuple[float, dict[str, Any]]:
    detected = str(parsed.get("detected_language", "")).lower()
    response_language = str(parsed.get("response_language", "")).lower()
    answer_label = str(parsed.get("answer_label", "")).strip().lower()
    translation = str(parsed.get("english_translation", "")).lower()
    expected_language = str(item.get("language", "")).lower()
    expected_response = str(item.get("expected_response_language", "")).lower()
    keyword_scores = [
        1.0 if translation_keyword_matches(str(keyword), translation) else 0.0
        for keyword in item.get("translation_keywords", [])
    ]
    language_score = 1.0 if language_matches(expected_language, detected) else 0.0
    response_score = 1.0 if language_matches(expected_response, response_language) else 0.0
    answer_score = 1.0 if answer_label == str(item.get("expected_answer_label", "")).lower() else 0.0
    translation_score = sum(keyword_scores) / len(keyword_scores) if keyword_scores else 0.0
    total = (language_score + response_score + answer_score + translation_score) / 4.0
    return total, {
        "language_score": language_score,
        "response_language_score": response_score,
        "answer_label_score": answer_score,
        "translation_keyword_score": translation_score,
    }


def language_matches(expected: str, observed: str) -> bool:
    expected_norm = expected.strip().lower()
    observed_norm = observed.strip().lower()
    if not expected_norm or not observed_norm:
        return False
    aliases = LANGUAGE_ALIASES.get(expected_norm, {expected_norm})
    return any(alias == observed_norm or alias in observed_norm for alias in aliases)


def translation_keyword_matches(keyword: str, translation: str) -> bool:
    key = keyword.strip().lower()
    normalized_translation = " ".join(translation.lower().split())
    alternatives = TRANSLATION_SYNONYMS.get(key, (key,))
    return any(alternative.lower() in normalized_translation for alternative in alternatives)


def format_review_context(rows: Sequence[Mapping[str, Any]], *, include_topic: bool, include_tags: bool) -> str:
    lines: list[str] = []
    for row in rows:
        parts = [
            f"id={row['id']}",
            f"rating={row.get('rating', '')}",
            f"title={row.get('title', '')}",
            f"review={row.get('content', '')}",
        ]
        if include_topic:
            parts.append(f"topic={row.get('topic', '')}")
        if include_tags:
            parts.append(f"semantic_tags={row.get('semantic_tags', '')}")
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def id_selection_prompt(agent_name: str, task: str, context: str) -> str:
    return (
        f"You are evaluating the {agent_name} in a product-review analysis workflow. "
        "Use only the provided fixed local review rows. Return JSON only with fields ids and answer. "
        "ids must be an array of integer review IDs that satisfy the task.\n"
        f"Task: {task}\n"
        f"Review rows:\n{context}"
    )


def parse_ids(parsed: Mapping[str, Any]) -> set[int]:
    raw = parsed.get("ids", parsed.get("evidence_ids", []))
    if isinstance(raw, str):
        values = re.findall(r"\d+", raw)
    elif isinstance(raw, Sequence):
        values = list(raw)
    else:
        values = []
    parsed_ids = {safe_int(value) for value in values}
    return {value for value in parsed_ids if value is not None}


def safe_int(value: Any) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def f1_score(expected: set[int], actual: set[int]) -> float:
    if not expected and not actual:
        return 1.0
    if not expected or not actual:
        return 0.0
    tp = len(expected.intersection(actual))
    precision = tp / len(actual) if actual else 0.0
    recall = tp / len(expected) if expected else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def masked_failure_mode(component: str) -> str:
    if component == "topic_assignment":
        return "topic_labels_masked"
    if component == "semantic_tags":
        return "semantic_tags_masked"
    return "restricted_context_error"


def score_vector_output(expected: set[int], expected_keywords: Sequence[str], parsed: Mapping[str, Any]) -> float:
    actual_ids = parse_ids(parsed)
    recall = len(expected.intersection(actual_ids)) / len(expected) if expected else 0.0
    text = " ".join(
        [
            str(parsed.get("answer", "")),
            " ".join(str(value) for value in parsed.get("evidence_snippets", []) if isinstance(parsed.get("evidence_snippets", []), list)),
        ]
    ).lower()
    if expected_keywords:
        keyword_score = sum(1 for keyword in expected_keywords if str(keyword).lower() in text) / len(expected_keywords)
    else:
        keyword_score = 0.0
    return max(0.0, min(1.0, 0.75 * recall + 0.25 * keyword_score))


def score_chart_spec(parsed: Mapping[str, Any], item: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> tuple[float, dict[str, Any]]:
    expected_chart = str(item.get("expected_chart_type", "")).lower()
    chart_type = str(parsed.get("chart_type", "")).lower()
    supported = parsed.get("supported")
    if expected_chart == "unsupported":
        unsupported_score = 1.0 if supported is False or chart_type == "unsupported" else 0.0
        return unsupported_score, {"unsupported_score": unsupported_score}
    group_by = str(parsed.get("group_by", "")).lower()
    aggregation = str(parsed.get("aggregation", "")).lower()
    y_field = str(parsed.get("y_field", "rating") or "rating").lower()
    if chart_type not in ALLOWED_CHART_TYPES:
        chart_type = ""
    if group_by not in ALLOWED_GROUP_FIELDS:
        group_by = ""
    if aggregation not in ALLOWED_AGGREGATIONS:
        aggregation = ""
    chart_score = 1.0 if chart_type == expected_chart else 0.0
    group_score = 1.0 if group_by == str(item.get("expected_group_by", "")).lower() else 0.0
    aggregation_score = 1.0 if aggregation == str(item.get("expected_aggregation", "")).lower() else 0.0
    value_score = score_expected_values(rows, group_by, aggregation, y_field, item.get("expected_values", {}))
    total = (chart_score + group_score + aggregation_score + value_score) / 4.0
    return total, {
        "chart_type_score": chart_score,
        "group_by_score": group_score,
        "aggregation_score": aggregation_score,
        "value_or_explanation_score": value_score,
        "normalized_spec": {
            "chart_type": chart_type,
            "group_by": group_by,
            "aggregation": aggregation,
            "y_field": y_field,
        },
    }


def score_expected_values(
    rows: Sequence[Mapping[str, Any]],
    group_by: str,
    aggregation: str,
    y_field: str,
    expected_values: Mapping[str, Any],
) -> float:
    if not expected_values:
        return 1.0
    if not group_by or not aggregation:
        return 0.0
    actual = aggregate_values(rows, group_by, aggregation, y_field)
    if set(actual) != {str(key) for key in expected_values}:
        return 0.0
    checks = []
    for key, expected in expected_values.items():
        actual_value = actual.get(str(key))
        try:
            expected_float = float(expected)
        except (TypeError, ValueError):
            checks.append(0.0)
            continue
        checks.append(1.0 if actual_value is not None and math.isclose(actual_value, expected_float, rel_tol=0.02, abs_tol=0.02) else 0.0)
    return sum(checks) / len(checks) if checks else 0.0


def aggregate_values(rows: Sequence[Mapping[str, Any]], group_by: str, aggregation: str, y_field: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        label = str(row.get(group_by, ""))
        if aggregation == "count":
            grouped[label].append(1.0)
        else:
            try:
                grouped[label].append(float(row.get(y_field, 0) or 0))
            except (TypeError, ValueError):
                grouped[label].append(0.0)
    if aggregation == "count":
        return {key: float(len(values)) for key, values in grouped.items()}
    if aggregation == "sum":
        return {key: float(sum(values)) for key, values in grouped.items()}
    return {key: float(sum(values) / len(values)) if values else 0.0 for key, values in grouped.items()}


def summarize_results(
    results: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    data: Mapping[str, Any],
    args: argparse.Namespace,
    preflight: Mapping[str, Any],
    components: Sequence[str] | None = None,
) -> dict[str, Any]:
    rows = [row for row in results if row.get("condition") in {"full", "restricted"}]
    summary_components = tuple(components or COMPONENT_ORDER)
    component_summaries: list[dict[str, Any]] = []
    table_ready_rows: list[dict[str, Any]] = []
    route_confusion: dict[str, dict[str, dict[str, int]]] = {"full": {}, "restricted": {}}
    for condition in ("full", "restricted"):
        matrix: dict[str, Counter[str]] = defaultdict(Counter)
        for row in rows:
            if row.get("component") == "orchestrator_routing" and row.get("condition") == condition:
                matrix[str(row.get("expected_route"))][str(row.get("actual_route"))] += 1
        route_confusion[condition] = {expected: dict(actuals) for expected, actuals in matrix.items()}

    for component in summary_components:
        component_rows = [row for row in rows if row.get("component") == component]
        full_rows = [row for row in component_rows if row.get("condition") == "full"]
        restricted_rows = [row for row in component_rows if row.get("condition") == "restricted"]
        full_success = mean([1.0 if row.get("success") else 0.0 for row in full_rows])
        restricted_success = mean([1.0 if row.get("success") else 0.0 for row in restricted_rows])
        full_quality = mean([float(row.get("quality_score") or 0.0) for row in full_rows])
        restricted_quality = mean([float(row.get("quality_score") or 0.0) for row in restricted_rows])
        quality_drop = full_quality - restricted_quality
        relative_drop = (quality_drop / full_quality * 100.0) if full_quality > 0 else 0.0
        failure_modes = Counter(str(row.get("failure_mode", "none")) for row in component_rows if row.get("failure_mode") != "none")
        workflow_calls = sum(int(row.get("workflow_calls") or 0) for row in component_rows)
        embedding_batches = sum(int(row.get("embedding_batches") or 0) for row in component_rows)
        summary_row = {
            "component": component,
            "component_label": COMPONENT_LABELS[component],
            "item_count": len({str(row.get("item_id")) for row in component_rows}),
            "full_success_rate": full_success,
            "restricted_success_rate": restricted_success,
            "success_drop": full_success - restricted_success,
            "full_quality_mean": full_quality,
            "restricted_quality_mean": restricted_quality,
            "quality_drop": quality_drop,
            "relative_quality_drop_percent": relative_drop,
            "workflow_calls": workflow_calls,
            "judge_calls": 0,
            "embedding_batches": embedding_batches,
            "failure_modes": dict(failure_modes),
            "main_observed_impact": observed_impact(full_quality, restricted_quality),
            "notes": component_notes(component, failure_modes),
        }
        component_summaries.append(summary_row)
        table_ready_rows.append(
            {
                key: summary_row[key]
                for key in (
                    "component",
                    "item_count",
                    "full_success_rate",
                    "restricted_success_rate",
                    "success_drop",
                    "full_quality_mean",
                    "restricted_quality_mean",
                    "quality_drop",
                    "relative_quality_drop_percent",
                    "workflow_calls",
                    "judge_calls",
                    "embedding_batches",
                    "main_observed_impact",
                    "notes",
                )
            }
        )

    token_summary = {
        "input_tokens": sum_optional(row.get("input_tokens") for row in rows),
        "output_tokens": sum_optional(row.get("output_tokens") for row in rows),
        "total_tokens": sum_optional(row.get("total_tokens") for row in rows),
        "workflow_calls": sum(int(row.get("workflow_calls") or 0) for row in rows),
        "judge_calls": 0,
        "embedding_batches": sum(int(row.get("embedding_batches") or 0) for row in rows),
        "api_error_rows": sum(1 for row in rows if row.get("api_error")),
    }
    matched_or_exceeded = [
        summary["component_label"]
        for summary in component_summaries
        if summary["restricted_quality_mean"] >= summary["full_quality_mean"]
    ]
    return {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "code_version": _git_commit_sha(),
        "model": args.model,
        "provider": args.provider,
        "semantic_backend": args.semantic_backend,
        "embedding_model": args.embedding_model,
        "items_path": str(Path(args.items).resolve()),
        "product_table": data.get("product_table"),
        "review_row_count": len(data["review_rows"]),
        "component_summaries": component_summaries,
        "table_ready_rows": table_ready_rows,
        "route_confusion_matrix": route_confusion,
        "token_summary": token_summary,
        "judge": {"used": False, "judge_calls": 0, "note": "No LLM judge used; deterministic scoring applied."},
        "components_where_restricted_matched_or_exceeded_full": matched_or_exceeded,
        "preflight": preflight,
        "selected_components": list(summary_components),
        "claim_boundary": (
            "Live controlled component-level ablation over fixed local inputs only; "
            "not evidence of large-scale robustness or population-level generalizability."
        ),
        "manuscript_status": "Not inserted into manuscript pending author review.",
    }


def observed_impact(full_quality: float, restricted_quality: float) -> str:
    if restricted_quality >= full_quality:
        return "Restricted condition matched or exceeded the full condition under this controlled item set."
    drop = full_quality - restricted_quality
    if drop >= 0.5:
        return "Large quality drop when the component was removed or restricted."
    if drop >= 0.2:
        return "Moderate quality drop when the component was removed or restricted."
    return "Small quality drop under the controlled item set."


def component_notes(component: str, failure_modes: Counter[str]) -> str:
    if not failure_modes:
        return "No failed rows recorded."
    common = ", ".join(f"{mode}: {count}" for mode, count in failure_modes.most_common(3))
    if component == "vector_retrieval":
        return f"Restricted baseline is static first-k evidence. Main failures: {common}."
    return f"Main failures: {common}."


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def sum_optional(values: Any) -> int | None:
    filtered = [int(value) for value in values if isinstance(value, (int, float))]
    return sum(filtered) if filtered else None


def append_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_summary_csv(path: Path, summaries: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "component",
        "component_label",
        "item_count",
        "full_success_rate",
        "restricted_success_rate",
        "success_drop",
        "full_quality_mean",
        "restricted_quality_mean",
        "quality_drop",
        "relative_quality_drop_percent",
        "workflow_calls",
        "judge_calls",
        "embedding_batches",
        "main_observed_impact",
        "notes",
    ]
    write_csv(path, summaries, fields)


def write_table_ready_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "component",
        "item_count",
        "full_success_rate",
        "restricted_success_rate",
        "success_drop",
        "full_quality_mean",
        "restricted_quality_mean",
        "quality_drop",
        "relative_quality_drop_percent",
        "workflow_calls",
        "judge_calls",
        "embedding_batches",
        "main_observed_impact",
        "notes",
    ]
    write_csv(path, rows, fields)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_summary_md(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# Live Controlled Component-Level Ablation Summary",
        "",
        f"Run ID: `{summary['run_id']}`",
        f"Model: `{summary['model']}` via `{summary['provider']}`",
        f"Semantic backend: `{summary['semantic_backend']}`",
        "",
        "Claim boundary: " + str(summary["claim_boundary"]),
        "",
        "| Component | Items | Full success | Restricted success | Full quality | Restricted quality | Quality drop | Calls | Embedding batches |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["component_summaries"]:
        lines.append(
            "| {component_label} | {item_count} | {full_success_rate:.3f} | {restricted_success_rate:.3f} | "
            "{full_quality_mean:.3f} | {restricted_quality_mean:.3f} | {quality_drop:.3f} | {workflow_calls} | {embedding_batches} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Route Confusion Matrix",
            "",
            "```json",
            json.dumps(summary["route_confusion_matrix"], indent=2),
            "```",
            "",
            "## Token And Call Summary",
            "",
            "```json",
            json.dumps(summary["token_summary"], indent=2),
            "```",
            "",
            "No LLM judge was used; deterministic scoring was applied to fixed expected labels, IDs, chart fields, and aggregations.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_failure_cases(path: Path, results: Sequence[Mapping[str, Any]]) -> None:
    failures = [row for row in results if row.get("condition") in {"full", "restricted"} and not row.get("success")]
    lines = ["# Live Controlled Component-Level Ablation Failure Cases", ""]
    if not failures:
        lines.append("No failed rows were recorded.")
    for row in failures:
        lines.extend(
            [
                f"## {row.get('item_id')} - {row.get('component_label')} - {row.get('condition')}",
                "",
                f"- Failure mode: `{row.get('failure_mode')}`",
                f"- Quality score: `{row.get('quality_score')}`",
                f"- Expected IDs: `{row.get('expected_ids')}`",
                f"- Actual IDs: `{row.get('actual_ids')}`",
                f"- Expected route: `{row.get('expected_route')}`",
                f"- Actual route: `{row.get('actual_route')}`",
                f"- Error: `{row.get('error')}`",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_judge_outputs(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "judge_used": False,
                "judge_calls": 0,
                "note": "No LLM-assisted judging was used. Scores are deterministic against fixed labels, IDs, chart specs, and aggregations.",
            }
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
