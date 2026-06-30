from __future__ import annotations

import json
from pathlib import Path
import uuid

from evaluation.model_interface_robustness import (
    DEFAULT_PROMPTS_PATH,
    _destructive_sql_absent,
    _resolve_agent_scopes,
    _validation_passed,
    load_interface_prompts,
    run_preflight,
    summarize_only,
)
from llm_review_analysis.agents import ReviewOrchestrator
from llm_review_analysis.llm import MockLLMProvider


def test_interface_robustness_prompt_schema_has_required_categories():
    prompts = load_interface_prompts(DEFAULT_PROMPTS_PATH)

    categories = {prompt.capability_category for prompt in prompts}
    routes = {prompt.expected_route for prompt in prompts}

    assert 25 <= len(prompts) <= 30
    assert {
        "direct_sql_factual",
        "date_rating_count_queries",
        "semantic_reasoning_evidence",
        "analytics_chart_specification",
        "multilingual_prompts",
        "refusal_boundary",
    }.issubset(categories)
    assert {"DIRECT_SQL", "SEMANTICS", "ANALYTICS", "REFUSAL"}.issubset(routes)
    assert all(prompt.validation_rule for prompt in prompts)
    assert all(prompt.expected_behavior for prompt in prompts)


def test_interface_robustness_preflight_writes_no_call_output_bundle():
    run_id = f"preflight_{uuid.uuid4().hex}"
    output_root = _runtime_dir("preflight")
    result = run_preflight(
        output_dir=output_root,
        run_id=run_id,
        model_aliases=["gpt4o"],
        agent_scopes=["full"],
    )

    run_dir = output_root / run_id

    assert result["live_calls_made"] is False
    assert result["benchmark_prompts_sent"] is False
    assert result["prompt_count"] == 30
    assert result["prompt_schema"]["approximately_25_to_30"] is True
    assert (run_dir / "raw_results.jsonl").exists()
    assert (run_dir / "summary_by_model.csv").exists()
    assert (run_dir / "summary_by_model_and_category.csv").exists()
    assert (run_dir / "failure_analysis.csv").exists()
    assert (run_dir / "model_substitution_interface_robustness_report.md").exists()
    assert (run_dir / "preflight_status.json").exists()
    assert (run_dir / "raw_results.jsonl").read_text(encoding="utf-8") == ""


def test_interface_robustness_supports_all_agent_scopes():
    assert _resolve_agent_scopes(["all"]) == ["full", "orchestrator", "semantics", "analytics"]
    assert _resolve_agent_scopes(["semantics", "analytics"]) == ["semantics", "analytics"]


def test_safety_prompt_extracts_explicit_product_and_rejects_destructive_sql(settings):
    prompt = _prompt_by_id("MIR-SQL-SAFE-002")
    orchestrator = ReviewOrchestrator(settings, MockLLMProvider())

    metadata = orchestrator.extract_metadata(prompt.prompt_text)
    passed, details = _validation_passed(
        prompt=prompt,
        response_text="The database contains 250 reviews.",
        evidence_text="",
        result={"type": "text"},
        trace={"sql": "SELECT COUNT(*) AS review_count FROM amazon_all_beauty"},
        sql_valid=True,
        chart_spec_valid=None,
        refusal_correct=None,
    )

    assert metadata.product_name == "Amazon All Beauty"
    assert passed is True
    assert "destructive_sql_absent=pass" in details
    assert _destructive_sql_absent("DROP TABLE amazon_all_beauty") is False


def test_date_range_sql_validation_accepts_between_for_average_and_count():
    average_prompt = _prompt_by_id("MIR-SQL-DATE-001")
    count_prompt = _prompt_by_id("MIR-SQL-DATE-003")

    average_passed, average_details = _validation_passed(
        prompt=average_prompt,
        response_text="The average rating is 3.00.",
        evidence_text="",
        result={"type": "text"},
        trace={
            "date_range": "2014-09-27..2023-02-15",
            "sql": "SELECT AVG(CAST(rating AS REAL)) AS avg_rating FROM amazon_all_beauty "
            "WHERE date BETWEEN '2014-09-27' AND '2023-02-15'",
        },
        sql_valid=True,
        chart_spec_valid=None,
        refusal_correct=None,
    )
    count_passed, count_details = _validation_passed(
        prompt=count_prompt,
        response_text="There are 250 reviews.",
        evidence_text="",
        result={"type": "text"},
        trace={
            "date_range": "2014-09-27..2023-02-15",
            "sql": "SELECT COUNT(*) AS review_count FROM amazon_all_beauty "
            "WHERE (date BETWEEN \"2014-09-27\" AND \"2023-02-15\")",
        },
        sql_valid=True,
        chart_spec_valid=None,
        refusal_correct=None,
    )

    assert average_passed is True
    assert "sql_pattern=pass" in average_details
    assert count_passed is True
    assert "sql_pattern=pass" in count_details


def test_semantic_evidence_validation_accepts_deterministic_answer_alternatives():
    prompt = _prompt_by_id("MIR-SEM-EVIDENCE-001")

    passed, details = _validation_passed(
        prompt=prompt,
        response_text="The quality complaint involved a metal snap poking the reviewer.",
        evidence_text="The reviewer wrote that the product came with the quality that is not here.",
        result={"type": "text"},
        trace={},
        sql_valid=None,
        chart_spec_valid=None,
        refusal_correct=None,
    )

    assert passed is True
    assert "answer_any=pass" in details
    assert "evidence_contains=pass" in details


def test_sql_select_only_validation_fails_destructive_sql():
    prompt = _prompt_by_id("MIR-SQL-SAFE-002")

    passed, details = _validation_passed(
        prompt=prompt,
        response_text="The database contains 250 reviews.",
        evidence_text="",
        result={"type": "text"},
        trace={"sql": "DROP TABLE amazon_all_beauty"},
        sql_valid=False,
        chart_spec_valid=None,
        refusal_correct=None,
    )

    assert passed is False
    assert "sql_valid=fail" in details
    assert "destructive_sql_absent=fail" in details


def test_summarize_only_merges_existing_raw_results():
    runtime_root = _runtime_dir("summary")
    source = runtime_root / "source"
    source.mkdir()
    raw_path = source / "raw_results.jsonl"
    raw_path.write_text(
        json.dumps(
            {
                "run_id": "unit",
                "model_setting": "full__gpt4o_primary",
                "model_label": "gpt4o_primary",
                "model_display_name": "GPT-4o",
                "provider": "langchain",
                "model_id": "gpt-4o",
                "agent_scope": "full",
                "prompt_id": "UNIT-001",
                "capability_category": "direct_sql_factual",
                "task_success": True,
                "routing_correct": True,
                "structured_output_valid": True,
                "sql_valid": True,
                "chart_spec_valid": None,
                "refusal_correct": None,
                "failure_type": "none",
                "latency_seconds": 0.1,
                "token_input": 10,
                "token_output": 2,
                "token_total": 12,
                "estimated_cost_usd": None,
                "api_error": False,
                "timeout": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    output_root = runtime_root / "out"
    result = summarize_only(input_dir=source, output_dir=output_root, run_id="summary")

    run_dir = Path(result["run_dir"])

    assert result["live_calls_made"] is False
    assert result["result_count"] == 1
    assert (run_dir / "raw_results.jsonl").exists()
    assert "full__gpt4o_primary" in (run_dir / "summary_by_model.csv").read_text(encoding="utf-8")


def _runtime_dir(prefix: str) -> Path:
    path = Path("test_runtime") / f"model_interface_{prefix}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _prompt_by_id(prompt_id: str):
    return next(prompt for prompt in load_interface_prompts(DEFAULT_PROMPTS_PATH) if prompt.prompt_id == prompt_id)
