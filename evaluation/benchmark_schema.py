from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROUTES = {"DIRECT_SQL", "SEMANTICS", "ANALYTICS"}
GOLD_VERIFICATION_STATUSES = {
    "author_verified",
    "programmatically_verified",
    "requires_author_input",
    "unverified",
    "verified",
}
GOLD_REQUIRED_FIELDS = {
    "prompt_id",
    "prompt_text",
    "language",
    "product",
    "expected_product_table",
    "expected_route",
    "expected_result_type",
    "success_criteria",
    "gold_verification_status",
}


@dataclass(frozen=True)
class BenchmarkPrompt:
    prompt_id: str
    prompt_text: str
    expected_route: str
    category: str = "general"
    language: str = "en"
    product: str | None = None
    expected_product_name: str | None = None
    expected_product: str | None = None
    expected_table: str | None = None
    expected_product_table: str | None = None
    expected_date_range: str | None = None
    expected_sql: str | None = None
    expected_sql_pattern: str | None = None
    expected_result_type: str | None = None
    expected_answer_facts: tuple[str, ...] = ()
    expected_answer_contains: tuple[str, ...] = ()
    expected_source_review_ids: tuple[str, ...] = ()
    expected_evidence_snippets: tuple[str, ...] = ()
    expected_evidence_contains: tuple[str, ...] = ()
    expected_chart_type: str | None = None
    expected_chart_group_by: str | None = None
    expected_chart_grouping: str | None = None
    expected_numeric_values: dict[str, float] = field(default_factory=dict)
    expected_chart_values: dict[str, float] = field(default_factory=dict)
    expected_failure_category: str | None = None
    expected_failure_type: str | None = None
    expected_refusal_category: str | None = None
    sql_eligible: bool | None = None
    sql_execution_eligible: bool | None = None
    chart_generation_eligible: bool | None = None
    chart_numeric_eligible: bool | None = None
    evidence_required: bool | None = None
    answer_fact_eligible: bool | None = None
    success_criteria: tuple[str, ...] = ()
    gold_verification_status: str = "unverified"
    gold_verified_by: str | None = None
    gold_computation_method: str | None = None
    gold_source_query: str | None = None
    gold_source_records: tuple[Any, ...] = ()
    gold_notes: str = ""
    evaluation_tags: tuple[str, ...] = ("benchmark", "baseline", "end-to-end")
    ambiguity_flag: bool = False
    missing_information_flag: bool = False
    contradiction_flag: bool = False
    multi_turn_context_flag: bool = False
    multiturn_context: bool = False

    @classmethod
    def from_mapping(cls, raw: dict[str, Any], *, require_gold_fields: bool = False) -> "BenchmarkPrompt":
        if require_gold_fields:
            missing = sorted(field for field in GOLD_REQUIRED_FIELDS if field not in raw)
            if missing:
                raise ValueError(f"Gold benchmark item {raw.get('prompt_id', '<unknown>')} is missing required field(s): {', '.join(missing)}")
        expected_route = str(raw["expected_route"]).upper()
        if expected_route not in ROUTES:
            raise ValueError(f"Unsupported expected_route for {raw.get('prompt_id')}: {expected_route}")
        gold_status = str(raw.get("gold_verification_status", "unverified")).strip().lower()
        if gold_status not in GOLD_VERIFICATION_STATUSES:
            raise ValueError(
                f"Unsupported gold_verification_status for {raw.get('prompt_id')}: {gold_status}"
            )
        expected_table = _optional_str(
            raw.get("expected_product_table", raw.get("expected_table", raw.get("expected_product")))
        )
        product_name = _optional_str(raw.get("product", raw.get("expected_product_name")))
        expected_chart_grouping = _optional_str(raw.get("expected_chart_grouping", raw.get("expected_chart_group_by")))
        expected_chart_values = _numeric_mapping(raw.get("expected_chart_values", raw.get("expected_numeric_values", {})))
        expected_failure = _optional_str(
            raw.get(
                "expected_refusal_category",
                raw.get("expected_failure_type", raw.get("expected_failure_category")),
            )
        )
        multi_turn = bool(raw.get("multiturn_context", raw.get("multi_turn_context_flag", False)))
        expected_route_for_eligibility = expected_route
        return cls(
            prompt_id=str(raw["prompt_id"]),
            prompt_text=str(raw["prompt_text"]),
            category=str(raw.get("category", "general")),
            language=str(raw.get("language", "en")),
            expected_route=expected_route,
            product=product_name,
            expected_product_name=product_name,
            expected_product=expected_table,
            expected_table=expected_table,
            expected_product_table=expected_table,
            expected_date_range=_optional_str(raw.get("expected_date_range")),
            expected_sql=_optional_str(raw.get("expected_sql")),
            expected_sql_pattern=_optional_str(raw.get("expected_sql_pattern")),
            expected_result_type=_optional_str(raw.get("expected_result_type")),
            expected_answer_facts=tuple(str(value) for value in raw.get("expected_answer_facts", ())),
            expected_answer_contains=tuple(str(value) for value in raw.get("expected_answer_contains", ())),
            expected_source_review_ids=tuple(str(value) for value in raw.get("expected_source_review_ids", ())),
            expected_evidence_snippets=tuple(str(value) for value in raw.get("expected_evidence_snippets", ())),
            expected_evidence_contains=tuple(str(value) for value in raw.get("expected_evidence_contains", ())),
            expected_chart_type=_optional_str(raw.get("expected_chart_type")),
            expected_chart_group_by=expected_chart_grouping,
            expected_chart_grouping=expected_chart_grouping,
            expected_numeric_values=expected_chart_values,
            expected_chart_values=expected_chart_values,
            expected_failure_category=expected_failure,
            expected_failure_type=expected_failure,
            expected_refusal_category=expected_failure,
            sql_eligible=_optional_bool(
                raw.get("sql_eligible"),
                default=expected_failure is None
                and bool(
                    expected_route_for_eligibility == "DIRECT_SQL"
                    or raw.get("expected_sql")
                    or raw.get("expected_sql_pattern")
                ),
            ),
            sql_execution_eligible=_optional_bool(
                raw.get("sql_execution_eligible"),
                default=expected_failure is None
                and bool(
                    expected_route_for_eligibility == "DIRECT_SQL"
                    or raw.get("expected_sql")
                    or raw.get("expected_sql_pattern")
                ),
            ),
            chart_generation_eligible=_optional_bool(
                raw.get("chart_generation_eligible"),
                default=expected_failure is None
                and expected_route_for_eligibility == "ANALYTICS"
                and _optional_str(raw.get("expected_result_type")) == "chart",
            ),
            chart_numeric_eligible=_optional_bool(
                raw.get("chart_numeric_eligible"),
                default=expected_failure is None and bool(expected_chart_values),
            ),
            evidence_required=_optional_bool(
                raw.get("evidence_required"),
                default=expected_failure is None
                and bool(
                    raw.get("expected_source_review_ids")
                    or raw.get("expected_evidence_snippets")
                    or raw.get("expected_evidence_contains")
                ),
            ),
            answer_fact_eligible=_optional_bool(
                raw.get("answer_fact_eligible"),
                default=expected_failure is None
                and bool(raw.get("expected_answer_facts") or raw.get("expected_answer_contains")),
            ),
            success_criteria=tuple(str(value) for value in raw.get("success_criteria", ())),
            gold_verification_status=gold_status,
            gold_verified_by=_optional_str(raw.get("gold_verified_by")),
            gold_computation_method=_optional_str(raw.get("gold_computation_method")),
            gold_source_query=_optional_str(raw.get("gold_source_query")),
            gold_source_records=tuple(raw.get("gold_source_records", ())),
            gold_notes=str(raw.get("gold_notes", "")),
            evaluation_tags=tuple(str(value) for value in raw.get("evaluation_tags", ("benchmark", "baseline", "end-to-end"))),
            ambiguity_flag=bool(raw.get("ambiguity_flag", False)),
            missing_information_flag=bool(raw.get("missing_information_flag", False)),
            contradiction_flag=bool(raw.get("contradiction_flag", False)),
            multi_turn_context_flag=multi_turn,
            multiturn_context=multi_turn,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkCheck:
    name: str
    passed: bool
    expected: Any = None
    actual: Any = None
    details: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkResult:
    run_id: str
    prompt_id: str
    prompt_text: str
    category: str
    language: str
    mode: str
    mode_name: str
    model_provider: str
    model: str
    expected_route: str
    actual_route: str | None
    expected_product: str | None
    actual_product: str | None
    expected_table: str | None
    actual_table: str | None
    expected_date_range: str | None
    actual_date_range: str | None
    expected_sql: str | None
    expected_sql_pattern: str | None
    actual_sql: str | None
    sql_valid: bool | None
    sql_execution_status: str
    expected_result_type: str | None
    actual_result_type: str | None
    expected_answer_facts: tuple[str, ...]
    expected_source_review_ids: tuple[str, ...]
    expected_evidence_snippets: tuple[str, ...]
    expected_evidence: tuple[str, ...]
    evidence_containment_status: bool | None
    expected_chart_type: str | None
    actual_chart_type: str | None
    expected_chart_grouping: str | None
    expected_chart_values: dict[str, float]
    actual_chart_values: dict[str, float]
    expected_numeric_values: dict[str, float]
    actual_numeric_values: dict[str, float]
    chart_numerical_consistency: bool | None
    expected_failure_type: str | None
    expected_refusal_category: str | None
    sql_eligible: bool | None
    sql_execution_eligible: bool | None
    chart_generation_eligible: bool | None
    chart_numeric_eligible: bool | None
    evidence_required: bool | None
    answer_fact_eligible: bool | None
    success_criteria: tuple[str, ...]
    gold_verification_status: str
    gold_verified_by: str | None
    gold_computation_method: str | None
    gold_source_query: str | None
    gold_source_records: tuple[Any, ...]
    gold_notes: str
    infrastructure_success: bool
    route_correctness: bool | None
    factual_correctness_proxy: bool | None
    evidence_containment: bool | None
    chart_structural_correctness: bool | None
    chart_numerical_correctness: bool | None
    expected_failure_handled: bool | None
    success: bool
    failure_category: str | None
    failure_reason: str | None
    latency_ms: float
    checks: tuple[BenchmarkCheck, ...] = ()
    chart_path: str | None = None
    chart_group_by: str | None = None
    evidence_ids: tuple[str, ...] = ()
    evidence_snippets: tuple[str, ...] = ()
    response_preview: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    estimated_cost_usd: float | None = None
    token_usage: dict[str, Any] | None = None
    failure_type: str | None = None
    failure_message: str | None = None
    mode_execution_type: str = "mock"
    uses_live_gpt4o: bool | None = None
    uses_faiss: bool | None = None
    uses_sql: bool | None = None
    uses_mock_provider: bool | None = None
    uses_deterministic_logic: bool | None = None
    disabled_enrichment_fields: tuple[str, ...] = ()
    inapplicable_reason: str | None = None
    live_call_count: int | None = None
    deterministic_call_count: int | None = None
    mock_call_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["checks"] = [check.to_dict() for check in self.checks]
        return payload


@dataclass(frozen=True)
class BenchmarkRunManifest:
    run_id: str
    mode: str
    modes: tuple[str, ...]
    dataset_name: str
    prompts_path: str
    reviews_path: str
    output_dir: str
    prompt_count: int
    result_count: int
    live_mode: bool
    model_provider: str
    model: str
    command: str
    evaluation_tags: tuple[str, ...]
    gold_schema_required: bool = False
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkArtifacts:
    run_id: str
    run_dir: Path
    manifest_path: Path
    results_path: Path
    summary_path: Path
    evidence_path: Path


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _numeric_mapping(value: Any) -> dict[str, float]:
    if not value:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("expected_numeric_values must be a JSON object")
    return {str(key): float(raw_value) for key, raw_value in value.items()}


def _optional_bool(value: Any, *, default: bool) -> bool | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return bool(value)

