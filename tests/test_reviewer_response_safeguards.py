from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re

import pytest

from llm_review_analysis.agents import AnalyticsAgent, ReviewOrchestrator
from llm_review_analysis.analytics import ChartSpec, ChartSpecValidationError, render_chart
from llm_review_analysis.db.sql_validator import SQLValidationError, validate_select_sql
from llm_review_analysis.llm import LLMResponse


def test_direct_sql_validator_accepts_read_only_select_with_approved_table_and_columns():
    sql = validate_select_sql(
        "SELECT rating, COUNT(*) AS review_count FROM sample_product GROUP BY rating ORDER BY rating",
        allowed_tables=["sample_product"],
    )

    assert sql == "SELECT rating, COUNT(*) AS review_count FROM sample_product GROUP BY rating ORDER BY rating"


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM sample_product",
        "UPDATE sample_product SET rating = '5'",
        "INSERT INTO sample_product (rating) VALUES ('5')",
        "REPLACE INTO sample_product (rating) VALUES ('5')",
        "CREATE TABLE copied_reviews AS SELECT * FROM sample_product",
        "ALTER TABLE sample_product ADD COLUMN secret TEXT",
        "DROP TABLE sample_product",
        "ATTACH DATABASE 'other.db' AS other",
        "VACUUM",
        "PRAGMA table_info(sample_product)",
        "SELECT COUNT(*) FROM sample_product; UPDATE sample_product SET rating = '5'",
    ],
)
def test_direct_sql_validator_rejects_write_schema_multi_statement_or_unsafe_sql(sql):
    with pytest.raises(SQLValidationError):
        validate_select_sql(sql, allowed_tables=["sample_product"])


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT COUNT(*) FROM other_product",
        "SELECT secret FROM sample_product",
        "SELECT rating, hidden_column FROM sample_product",
    ],
)
def test_direct_sql_validator_rejects_unapproved_tables_or_columns(sql):
    with pytest.raises(SQLValidationError):
        validate_select_sql(sql, allowed_tables=["sample_product"])


def test_direct_sql_route_falls_back_when_llm_proposes_unsafe_sql(settings, sample_db):
    conn, table = sample_db
    guarded_settings = replace(settings, llm_provider="langchain", allow_live_llm=True)
    orchestrator = ReviewOrchestrator(guarded_settings, _UnsafeSqlProvider())

    result, trace = orchestrator.answer_with_trace(conn, "How many reviews for sample product?", product_table=table)

    assert result["type"] == "text"
    assert "2 reviews" in result["message"]
    assert trace["sql_planner"] == "deterministic_fallback_after_live_sql"
    assert trace["sql"].startswith("SELECT COUNT(*)")
    assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 2


def test_chart_spec_accepts_constrained_supported_specification():
    spec = ChartSpec.from_mapping(
        {
            "chart_type": "bar",
            "x_field": "rating",
            "aggregation": "count",
            "group_by": "rating",
            "title": "Rating distribution",
        }
    )

    assert spec.chart_type == "bar"
    assert spec.x_field == "rating"
    assert spec.aggregation == "count"
    assert spec.group_by == "rating"


@pytest.mark.parametrize(
    "raw",
    [
        {"chart_type": "scatter", "x_field": "rating", "aggregation": "count"},
        {"chart_type": "bar", "x_field": "secret", "aggregation": "count"},
        {"chart_type": "bar", "x_field": "rating", "aggregation": "median"},
        {"chart_type": "line", "x_field": "date", "aggregation": "avg", "group_by": "date"},
        {"chart_type": "bar", "x_field": "rating", "aggregation": "count", "group_by": "secret"},
    ],
)
def test_chart_spec_rejects_invalid_chart_type_fields_aggregation_or_grouping(raw):
    with pytest.raises(ChartSpecValidationError):
        ChartSpec.from_mapping(raw)


def test_analytics_rejects_unsupported_chart_type_with_controlled_response(settings, provider, sample_db):
    conn, table = sample_db

    result = AnalyticsAgent(settings, provider).run(conn, table, "Create a heatmap of ratings by country")

    assert result["type"] == "text"
    assert result["failure_category"] == "unsupported_chart_type"
    assert "supported chart types" in result["message"].lower()


def test_analytics_rejects_underspecified_chart_request_with_controlled_response(settings, provider, sample_db):
    conn, table = sample_db

    result = AnalyticsAgent(settings, provider).run(conn, table, "Show me a chart for sample product")

    assert result["type"] == "text"
    assert result["failure_category"] == "ambiguous_analytics_request"
    assert "underspecified" in result["message"].lower()


def test_supported_charts_render_through_png_renderer(settings):
    spec = ChartSpec.from_mapping(
        {
            "chart_type": "line",
            "x_field": "date",
            "y_field": "rating",
            "aggregation": "avg",
            "group_by": "date",
            "title": "Average rating over time",
        }
    )

    path = render_chart(spec, [("2024-01-01", 4.0), ("2024-01-02", 3.5)], settings.output_dir, stem="guardrail_line")

    assert path.suffix == ".png"
    assert path.read_bytes().startswith(b"\x89PNG")


def test_no_generated_plotting_code_execution_path_is_present():
    implementation_paths = [
        Path("src/llm_review_analysis/agents/analytics_agent.py"),
        Path("src/llm_review_analysis/analytics/chart_renderer.py"),
        Path("src/llm_review_analysis/analytics/chart_specs.py"),
    ]
    implementation_text = "\n".join(path.read_text(encoding="utf-8") for path in implementation_paths)

    assert re.search(r"\bexec\s*\(", implementation_text) is None
    assert re.search(r"\beval\s*\(", implementation_text) is None
    assert "exec_globals" not in implementation_text
    assert "subprocess" not in implementation_text
    assert "shell=True" not in implementation_text


def test_legacy_unsafe_analytics_behavior_is_not_reintroduced():
    implementation_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in Path("src/llm_review_analysis").rglob("*.py")
    )

    assert "revision_materials/legacy_implementation" not in implementation_text
    assert "revision_materials\\legacy_implementation" not in implementation_text
    assert "exec_globals" not in implementation_text
    assert re.search(r"\bexec\s*\(", implementation_text) is None


def test_unsupported_route_output_still_reaches_controlled_refusal(settings, sample_db):
    conn, table = sample_db
    orchestrator = ReviewOrchestrator(settings, _UnsupportedRouteProvider())

    result, trace = orchestrator.answer_with_trace(
        conn,
        "Show a scatter plot of rating by date for sample product",
        product_table=table,
    )

    assert trace["route"] == "ANALYTICS"
    assert result["type"] == "text"
    assert result["failure_category"] == "unsupported_chart_type"
    assert trace["failure_category"] == "unsupported_chart_type"


class _UnsafeSqlProvider:
    model = "unsafe-sql-provider"

    def generate(self, prompt: str, *, purpose: str = "general", response_format: str | None = None) -> LLMResponse:
        if purpose == "route":
            return LLMResponse(content="DIRECT_SQL", model=self.model)
        if purpose == "sql_generation":
            return LLMResponse(content='{"sql":"DROP TABLE sample_product"}', model=self.model)
        return LLMResponse(content="{}", model=self.model)


class _UnsupportedRouteProvider:
    model = "unsupported-route-provider"

    def generate(self, prompt: str, *, purpose: str = "general", response_format: str | None = None) -> LLMResponse:
        if purpose == "route":
            return LLMResponse(content="UNSUPPORTED_ROUTE", model=self.model)
        if purpose == "chart_spec":
            return LLMResponse(
                content='{"chart_type":"bar","x_field":"rating","aggregation":"count","group_by":"rating"}',
                model=self.model,
            )
        return LLMResponse(content="{}", model=self.model)
