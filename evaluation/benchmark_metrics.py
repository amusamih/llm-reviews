from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Any, Iterable

from evaluation.benchmark_schema import BenchmarkResult


def summarize_benchmark_results(results: Iterable[BenchmarkResult]) -> dict[str, Any]:
    result_list = list(results)
    aggregate = _summarize_group(result_list)
    by_mode = {
        mode: _summarize_group(mode_results)
        for mode, mode_results in sorted(_group_by_mode(result_list).items())
    }
    by_gold_status = {
        status: _summarize_group(status_results)
        for status, status_results in sorted(_group_by_gold_status(result_list).items())
    }
    aggregate["by_mode"] = by_mode
    aggregate["by_gold_verification_status"] = by_gold_status
    aggregate["verified_only"] = _summarize_group(
        [result for result in result_list if result.gold_verification_status in {"author_verified", "verified"}]
    )
    aggregate["programmatically_verified_only"] = _summarize_group(
        [result for result in result_list if result.gold_verification_status == "programmatically_verified"]
    )
    aggregate["verified_or_programmatically_verified"] = _summarize_group(
        [
            result
            for result in result_list
            if result.gold_verification_status in {"author_verified", "programmatically_verified", "verified"}
        ]
    )
    aggregate["unverified_or_requires_author_input"] = _summarize_group(
        [
            result
            for result in result_list
            if result.gold_verification_status in {"requires_author_input", "unverified"}
        ]
    )
    aggregate["mode_count"] = len(by_mode)
    aggregate["gold_verification_status_counts"] = dict(
        sorted(Counter(result.gold_verification_status for result in result_list).items())
    )
    return aggregate


def _summarize_group(result_list: list[BenchmarkResult]) -> dict[str, Any]:
    prompt_count = len(result_list)
    check_counts: dict[str, Counter[str]] = {}
    failure_categories: Counter[str] = Counter()
    latencies = [result.latency_ms for result in result_list]

    for result in result_list:
        if result.failure_category:
            failure_categories[result.failure_category] += 1
        elif result.failure_type:
            failure_categories[result.failure_type] += 1
        for check in result.checks:
            counter = check_counts.setdefault(check.name, Counter())
            counter["total"] += 1
            if check.passed:
                counter["passed"] += 1
            else:
                counter["failed"] += 1

    checks = {
        name: {
            "passed": counts["passed"],
            "failed": counts["failed"],
            "total": counts["total"],
            "rate": _safe_divide(counts["passed"], counts["total"]),
        }
        for name, counts in sorted(check_counts.items())
    }
    success_count = sum(1 for result in result_list if result.success)
    failure_count = sum(1 for result in result_list if not result.success)
    inapplicable_count = sum(
        1
        for result in result_list
        if result.failure_category == "inapplicable_for_mode" or result.inapplicable_reason
    )
    raw_aggregate_metrics = _raw_aggregate_metrics(checks)
    eligibility_adjusted_metrics = _eligibility_adjusted_metrics(result_list)
    expected_refusal_metrics = _expected_refusal_metrics(result_list)
    return {
        "prompt_count": prompt_count,
        "applicable_prompt_count": prompt_count - inapplicable_count,
        "inapplicable_count": inapplicable_count,
        "inapplicable_rate": _safe_divide(inapplicable_count, prompt_count),
        "success_count": success_count,
        "failure_count": failure_count,
        "infrastructure_success_rate": _bool_rate(
            result.infrastructure_success for result in result_list
        ),
        "overall_success_rate": _safe_divide(success_count, prompt_count),
        "failure_rate": _safe_divide(failure_count, prompt_count),
        "routing_accuracy": _check_rate(checks, "route"),
        "route_correctness_rate": _bool_rate(
            result.route_correctness for result in result_list
        ),
        "product_table_extraction_accuracy": _check_rate(checks, "product_table"),
        "date_range_extraction_accuracy": _check_rate(checks, "date_range"),
        "sql_validity_rate": _check_rate(checks, "sql_valid"),
        "sql_execution_success_rate": _check_rate(checks, "sql_execution"),
        "answer_containment_factuality_proxy_rate": _combined_check_rate(checks, ("answer_contains", "evidence_contains")),
        "factual_correctness_proxy_rate": _bool_rate(
            result.factual_correctness_proxy for result in result_list
        ),
        "answer_fact_accuracy": _check_rate(checks, "answer_facts"),
        "evidence_containment_rate": _check_rate(checks, "evidence_contains"),
        "evidence_containment_status_rate": _bool_rate(
            result.evidence_containment for result in result_list
        ),
        "source_review_id_accuracy": _check_rate(checks, "source_review_ids"),
        "chart_type_accuracy": _check_rate(checks, "chart_type"),
        "chart_file_generation_rate": _check_rate(checks, "chart_file_exists"),
        "chart_structural_correctness_rate": _bool_rate(
            result.chart_structural_correctness for result in result_list
        ),
        "chart_numerical_consistency_rate": _check_rate(checks, "chart_numeric_values"),
        "chart_numerical_correctness_rate": _bool_rate(
            result.chart_numerical_correctness for result in result_list
        ),
        "expected_failure_handling_rate": _bool_rate(
            result.expected_failure_handled for result in result_list
        ),
        "overall_success_rate_with_expected_refusals": _safe_divide(success_count, prompt_count),
        "sql_validity_rate_sql_eligible": eligibility_adjusted_metrics["sql_validity_rate"]["rate"],
        "sql_execution_success_rate_sql_executable": eligibility_adjusted_metrics["sql_execution_success_rate"]["rate"],
        "chart_file_generation_rate_chart_generation_eligible": eligibility_adjusted_metrics["chart_file_generation_rate"]["rate"],
        "chart_numerical_consistency_rate_chart_numeric_eligible": eligibility_adjusted_metrics["chart_numerical_consistency_rate"]["rate"],
        "evidence_containment_rate_evidence_required": eligibility_adjusted_metrics["evidence_containment_rate"]["rate"],
        "answer_fact_proxy_rate_answer_fact_eligible": eligibility_adjusted_metrics["answer_fact_proxy_rate"]["rate"],
        "raw_aggregate_metrics": raw_aggregate_metrics,
        "eligibility_adjusted_metrics": eligibility_adjusted_metrics,
        "expected_refusal_metrics": expected_refusal_metrics,
        "metric_denominators": {
            "raw": {
                name: metric["denominator"]
                for name, metric in raw_aggregate_metrics.items()
            },
            "eligibility_adjusted": {
                name: metric["denominator"]
                for name, metric in eligibility_adjusted_metrics.items()
            },
            "expected_refusal": {
                name: metric["denominator"]
                for name, metric in expected_refusal_metrics.items()
            },
        },
        "checks": checks,
        "latency_ms": latency_summary(latencies),
        "mode_execution_type_counts": dict(
            sorted(Counter(result.mode_execution_type for result in result_list).items())
        ),
        "live_call_count": sum(int(result.live_call_count or 0) for result in result_list),
        "mock_call_count": sum(int(result.mock_call_count or 0) for result in result_list),
        "deterministic_call_count": sum(
            int(result.deterministic_call_count or 0) for result in result_list
        ),
        "gold_verification_status_counts": dict(
            sorted(Counter(result.gold_verification_status for result in result_list).items())
        ),
        "failure_rate_by_category": {
            category: _safe_divide(count, prompt_count)
            for category, count in sorted(failure_categories.items())
        },
        "failure_categories": dict(sorted(failure_categories.items())),
        "failure_types": dict(sorted(failure_categories.items())),
    }


def _raw_aggregate_metrics(checks: dict[str, dict[str, float | int]]) -> dict[str, dict[str, float | int | None]]:
    return {
        "sql_validity_rate": _raw_check_metric(checks, "sql_valid"),
        "sql_execution_success_rate": _raw_check_metric(checks, "sql_execution"),
        "answer_fact_accuracy": _raw_check_metric(checks, "answer_facts"),
        "evidence_containment_rate": _raw_check_metric(checks, "evidence_contains"),
        "evidence_snippet_containment_rate": _raw_check_metric(checks, "evidence_snippets"),
        "source_review_id_accuracy": _raw_check_metric(checks, "source_review_ids"),
        "chart_file_generation_rate": _raw_check_metric(checks, "chart_file_exists"),
        "chart_numerical_consistency_rate": _raw_check_metric(checks, "chart_numeric_values"),
    }


def _raw_check_metric(
    checks: dict[str, dict[str, float | int]],
    name: str,
) -> dict[str, float | int | None]:
    if name not in checks:
        return {"passed": 0, "denominator": 0, "rate": None}
    return {
        "passed": int(checks[name]["passed"]),
        "denominator": int(checks[name]["total"]),
        "rate": float(checks[name]["rate"]),
    }


def _eligibility_adjusted_metrics(result_list: list[BenchmarkResult]) -> dict[str, dict[str, float | int | None]]:
    return {
        "sql_validity_rate": _eligible_metric(
            result_list,
            eligible=lambda result: result.sql_eligible is True,
            passed=lambda result: _result_check_passed(result, "sql_valid"),
        ),
        "sql_execution_success_rate": _eligible_metric(
            result_list,
            eligible=lambda result: result.sql_execution_eligible is True,
            passed=lambda result: _result_check_passed(result, "sql_execution"),
        ),
        "chart_file_generation_rate": _eligible_metric(
            result_list,
            eligible=lambda result: result.chart_generation_eligible is True,
            passed=lambda result: _result_check_passed(result, "chart_file_exists"),
        ),
        "chart_numerical_consistency_rate": _eligible_metric(
            result_list,
            eligible=lambda result: result.chart_numeric_eligible is True,
            passed=lambda result: _result_check_passed(result, "chart_numeric_values"),
        ),
        "evidence_containment_rate": _eligible_metric(
            result_list,
            eligible=lambda result: result.evidence_required is True,
            passed=lambda result: result.evidence_containment is True,
        ),
        "answer_fact_proxy_rate": _eligible_metric(
            result_list,
            eligible=lambda result: result.answer_fact_eligible is True,
            passed=lambda result: result.factual_correctness_proxy is True,
        ),
    }


def _expected_refusal_metrics(result_list: list[BenchmarkResult]) -> dict[str, dict[str, float | int | None]]:
    return {
        "ambiguity_graceful_refusal_rate": _refusal_metric(result_list, ("ambiguous_prompt",)),
        "missing_information_graceful_refusal_rate": _refusal_metric(result_list, ("missing_information",)),
        "unknown_product_graceful_refusal_rate": _refusal_metric(
            result_list,
            ("product_not_found", "unknown_product", "table_mismatch"),
        ),
        "unsupported_chart_graceful_refusal_rate": _refusal_metric(result_list, ("unsupported_chart_type",)),
        "translation_quality_not_evaluated_refusal_rate": _refusal_metric(
            result_list,
            ("translation_quality_not_evaluated",),
        ),
        "overall_expected_failure_handling_rate": _eligible_metric(
            result_list,
            eligible=lambda result: result.expected_refusal_category is not None,
            passed=lambda result: result.expected_failure_handled is True,
        ),
    }


def _eligible_metric(
    result_list: list[BenchmarkResult],
    *,
    eligible: Any,
    passed: Any,
) -> dict[str, float | int | None]:
    eligible_results = [result for result in result_list if eligible(result)]
    denominator = len(eligible_results)
    passed_count = sum(1 for result in eligible_results if passed(result))
    return {
        "passed": passed_count,
        "denominator": denominator,
        "rate": None if denominator == 0 else _safe_divide(passed_count, denominator),
    }


def _refusal_metric(
    result_list: list[BenchmarkResult],
    categories: tuple[str, ...],
) -> dict[str, float | int | None]:
    category_set = set(categories)
    return _eligible_metric(
        result_list,
        eligible=lambda result: result.expected_refusal_category in category_set,
        passed=lambda result: result.expected_failure_handled is True,
    )


def _result_check_passed(result: BenchmarkResult, name: str) -> bool:
    return any(check.name == name and check.passed for check in result.checks)


def latency_summary(values: Iterable[float]) -> dict[str, float | int]:
    latencies = sorted(float(value) for value in values)
    if not latencies:
        return {
            "count": 0,
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }
    return {
        "count": len(latencies),
        "mean": round(mean(latencies), 3),
        "min": round(latencies[0], 3),
        "max": round(latencies[-1], 3),
        "p50": round(_percentile(latencies, 50), 3),
        "p95": round(_percentile(latencies, 95), 3),
        "p99": round(_percentile(latencies, 99), 3),
    }


def contains_all(text: str, terms: Iterable[str]) -> bool:
    lower = text.lower()
    return all(term.lower() in lower for term in terms)


def normalize_sql(sql: str | None) -> str:
    if not sql:
        return ""
    return " ".join(sql.strip().rstrip(";").split()).lower()


def _group_by_mode(results: list[BenchmarkResult]) -> dict[str, list[BenchmarkResult]]:
    grouped: dict[str, list[BenchmarkResult]] = defaultdict(list)
    for result in results:
        grouped[result.mode].append(result)
    return grouped


def _group_by_gold_status(results: list[BenchmarkResult]) -> dict[str, list[BenchmarkResult]]:
    grouped: dict[str, list[BenchmarkResult]] = defaultdict(list)
    for result in results:
        grouped[result.gold_verification_status].append(result)
    return grouped


def _check_rate(checks: dict[str, dict[str, float | int]], name: str) -> float | None:
    if name not in checks:
        return None
    return float(checks[name]["rate"])


def _combined_check_rate(checks: dict[str, dict[str, float | int]], names: tuple[str, ...]) -> float | None:
    passed = 0
    total = 0
    for name in names:
        if name not in checks:
            continue
        passed += int(checks[name]["passed"])
        total += int(checks[name]["total"])
    if total == 0:
        return None
    return _safe_divide(passed, total)


def _bool_rate(values: Iterable[bool | None]) -> float | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    passed = sum(1 for value in filtered if value)
    return _safe_divide(passed, len(filtered))


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (percentile / 100.0) * (len(sorted_values) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = rank - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


def _safe_divide(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 6)
