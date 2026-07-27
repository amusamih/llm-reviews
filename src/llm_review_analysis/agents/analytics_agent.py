from __future__ import annotations

import base64
import hashlib
import json
import re
import sqlite3
from pathlib import Path

from llm_review_analysis.analytics import ChartSpec, render_chart
from llm_review_analysis.analytics.chart_specs import CHART_TYPES
from llm_review_analysis.config import Settings
from llm_review_analysis.db.schema import REVIEW_COLUMNS, validate_identifier
from llm_review_analysis.db.sql_validator import execute_validated_select
from llm_review_analysis.llm import LLMProvider


class AnalyticsAgent:
    def __init__(self, settings: Settings, provider: LLMProvider) -> None:
        self.settings = settings
        self.provider = provider

    def run(self, conn: sqlite3.Connection, table_name: str, prompt: str) -> dict[str, object]:
        table = validate_identifier(table_name)
        unsupported_chart = _requested_unsupported_chart_type(prompt)
        if unsupported_chart:
            return {
                "type": "text",
                "message": (
                    f"Unsupported chart type requested: {unsupported_chart}. "
                    f"Supported chart types are: {', '.join(sorted(CHART_TYPES))}."
                ),
                "failure_category": "unsupported_chart_type",
                "failure_reason": f"Requested chart type is not supported: {unsupported_chart}.",
            }
        if _is_underspecified_analytics_request(prompt):
            return {
                "type": "text",
                "message": (
                    "The analytics request is underspecified. Please specify the field, grouping, "
                    "or aggregation to visualize, such as rating distribution, average rating by date, "
                    "or review counts by country."
                ),
                "failure_category": "ambiguous_analytics_request",
                "failure_reason": "No supported analytics field, grouping, or aggregation was specified.",
            }
        spec = self._build_spec(prompt)
        sql = build_aggregate_sql(table, spec, prompt=prompt)
        _, rows = execute_validated_select(conn, sql, allowed_tables=[table], allowed_columns=REVIEW_COLUMNS)
        image_path = render_chart(spec, rows, self.settings.output_dir, stem=_chart_stem(prompt))
        image_b64 = _encode_file(image_path)
        chart_rows = [{"label": str(row[0]), "value": float(row[1] or 0)} for row in rows]
        return {
            "type": "chart",
            "path": str(image_path),
            "base64": image_b64,
            "chart_type": spec.chart_type,
            "aggregation": spec.aggregation,
            "x_field": spec.x_field,
            "y_field": spec.y_field or "",
            "group_by": spec.group_by or spec.x_field,
            "chart_rows": chart_rows,
            "explanation": _explain_chart(spec, rows),
        }

    def _build_spec(self, prompt: str) -> ChartSpec:
        policy_spec = _policy_spec(prompt)
        if policy_spec:
            return policy_spec
        response = self.provider.generate(_chart_spec_prompt(prompt), purpose="chart_spec", response_format="json")
        try:
            raw = json.loads(response.content)
            return _apply_chart_selection_policy(prompt, ChartSpec.from_mapping(raw))
        except Exception:
            return _fallback_spec(prompt)


def build_aggregate_sql(table: str, spec: ChartSpec, *, prompt: str = "") -> str:
    x_field = validate_identifier(spec.group_by or spec.x_field)
    where_clause = _analytics_where_clause(prompt)
    if spec.aggregation == "count":
        return f"SELECT {x_field}, COUNT(*) AS value FROM {table}{where_clause} GROUP BY {x_field} ORDER BY {x_field}"
    y_field = validate_identifier(spec.y_field or "rating")
    expression = "AVG" if spec.aggregation == "avg" else "SUM"
    return f"SELECT {x_field}, {expression}(CAST({y_field} AS REAL)) AS value FROM {table}{where_clause} GROUP BY {x_field} ORDER BY {x_field}"


def _analytics_where_clause(prompt: str) -> str:
    clauses: list[str] = []
    date_clause = _analytics_date_clause(prompt)
    if date_clause:
        clauses.append(date_clause)
    rating = _analytics_rating_filter(prompt)
    if rating is not None:
        clauses.append(f"CAST(rating AS REAL) = {rating}")
    sentiment = _analytics_sentiment_filter(prompt)
    if sentiment:
        clauses.append(f"LOWER(semantic_tags) LIKE '%{sentiment}%'")
    topic = _analytics_topic_filter(prompt)
    if topic:
        escaped = topic.replace("'", "''").lower()
        clauses.append(f"LOWER(topic) LIKE '%{escaped}%'")
    return " WHERE " + " AND ".join(clauses) if clauses else ""


def _analytics_rating_filter(prompt: str) -> int | None:
    lower = prompt.lower()
    word_to_rating = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
    for word, value in word_to_rating.items():
        if re.search(rf"\b{word}\s*[- ]?star\b", lower):
            return value
    match = re.search(r"\b([1-5])\s*[- ]?star\b", lower)
    if match:
        return int(match.group(1))
    match = re.search(r"\brating\s*(?:=|is|of)?\s*([1-5])\b", lower)
    if match:
        return int(match.group(1))
    return None


def _analytics_sentiment_filter(prompt: str) -> str | None:
    lower = prompt.lower()
    for sentiment in ("negative", "positive", "neutral"):
        if re.search(rf"\b{sentiment}\b", lower):
            return sentiment
    return None


def _analytics_topic_filter(prompt: str) -> str | None:
    match = re.search(r"\btopic\s+(?:is|=|about)?\s*([a-z][a-z0-9 _-]{2,30})", prompt.lower())
    if not match:
        return None
    return re.sub(r"\s+", " ", match.group(1)).strip(" .?!")


def _analytics_date_clause(prompt: str) -> str:
    date_pattern = r"\d{4}-\d{2}-\d{2}"
    between_match = re.search(rf"\bbetween\s+({date_pattern})\s+and\s+({date_pattern})\b", prompt, flags=re.IGNORECASE)
    if between_match:
        return f"date >= '{between_match.group(1)}' AND date <= '{between_match.group(2)}'"
    from_match = re.search(rf"\bfrom\s+({date_pattern})\s+to\s+({date_pattern})\b", prompt, flags=re.IGNORECASE)
    if from_match:
        return f"date >= '{from_match.group(1)}' AND date <= '{from_match.group(2)}'"
    on_match = re.search(rf"\bon\s+({date_pattern})\b", prompt, flags=re.IGNORECASE)
    if on_match:
        return f"date = '{on_match.group(1)}'"
    return ""

def _chart_spec_prompt(prompt: str) -> str:
    return (
        "Return a JSON chart specification for the review database. "
        "Allowed chart_type values are bar, line, and pie. "
        "Use bar for rating distributions or ordinal/numeric categories; line for date/time trends; "
        "pie only for small nominal part-to-whole share/percentage requests. "
        "Allowed fields are chart_type, x_field, y_field, aggregation, group_by, title, x_label, y_label, language. "
        f"User prompt: {prompt}"
    )


def _fallback_spec(prompt: str) -> ChartSpec:
    policy_spec = _policy_spec(prompt)
    if policy_spec:
        return policy_spec
    lower = prompt.lower()
    if _mentions_country(lower):
        return ChartSpec.from_mapping(
            {
                "chart_type": "bar",
                "x_field": "country",
                "aggregation": "count",
                "group_by": "country",
                "title": "Reviews by country",
                "x_label": "Country",
                "y_label": "Count",
            }
        )
    if _mentions_time_trend(lower):
        return ChartSpec.from_mapping(
            {
                "chart_type": "line",
                "x_field": "date",
                "y_field": "rating",
                "aggregation": "avg",
                "group_by": "date",
                "title": "Average rating over time",
                "x_label": "Date",
                "y_label": "Average rating",
            }
        )
    return ChartSpec.from_mapping(
        {
            "chart_type": "bar",
            "x_field": "rating",
            "aggregation": "count",
            "group_by": "rating",
            "title": "Review rating distribution",
            "x_label": "Rating",
            "y_label": "Count",
        }
    )


def _apply_chart_selection_policy(prompt: str, spec: ChartSpec) -> ChartSpec:
    policy_spec = _policy_spec(prompt)
    if policy_spec:
        return policy_spec
    return spec


def _policy_spec(prompt: str) -> ChartSpec | None:
    lower = prompt.lower()
    explicit_chart = _requested_supported_chart_type(lower)
    if explicit_chart:
        return _spec_for_explicit_chart(lower, explicit_chart)
    if _mentions_rating_distribution(lower):
        return _rating_distribution_spec("bar")
    if _mentions_time_trend(lower):
        return _date_trend_spec("line")
    if _mentions_share_request(lower) and _mentions_country(lower):
        return _country_share_spec("pie")
    return None


def _spec_for_explicit_chart(lower_prompt: str, chart_type: str) -> ChartSpec:
    if _mentions_time_trend(lower_prompt):
        return _date_trend_spec(chart_type)
    if _mentions_country(lower_prompt):
        return _country_share_spec(chart_type)
    return _rating_distribution_spec(chart_type)


def _rating_distribution_spec(chart_type: str) -> ChartSpec:
    return ChartSpec.from_mapping(
        {
            "chart_type": chart_type,
            "x_field": "rating",
            "aggregation": "count",
            "group_by": "rating",
            "title": "Review rating distribution",
            "x_label": "Rating",
            "y_label": "Count",
        }
    )


def _date_trend_spec(chart_type: str) -> ChartSpec:
    return ChartSpec.from_mapping(
        {
            "chart_type": chart_type,
            "x_field": "date",
            "y_field": "rating",
            "aggregation": "avg",
            "group_by": "date",
            "title": "Average rating over time",
            "x_label": "Date",
            "y_label": "Average rating",
        }
    )


def _country_share_spec(chart_type: str) -> ChartSpec:
    return ChartSpec.from_mapping(
        {
            "chart_type": chart_type,
            "x_field": "country",
            "aggregation": "count",
            "group_by": "country",
            "title": "Reviews by country",
            "x_label": "Country",
            "y_label": "Count",
        }
    )


def _encode_file(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _explain_chart(spec: ChartSpec, rows: list[tuple]) -> str:
    return f"Generated a {spec.chart_type} chart with {len(rows)} grouped values using {spec.aggregation} over {spec.x_field}."


def _chart_stem(prompt: str) -> str:
    digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]
    return f"chart_{digest}"


SUPPORTED_CHART_TERMS = {
    "bar": "bar",
    "bar chart": "bar",
    "line": "line",
    "line chart": "line",
    "pie": "pie",
    "pie chart": "pie",
}


UNSUPPORTED_CHART_TERMS = {
    "area",
    "box",
    "boxplot",
    "bubble",
    "heatmap",
    "histogram",
    "radar",
    "scatter",
    "treemap",
    "violin",
}


def _requested_supported_chart_type(prompt_lower: str) -> str | None:
    for term, chart_type in sorted(SUPPORTED_CHART_TERMS.items(), key=lambda item: len(item[0]), reverse=True):
        if re.search(rf"\b{re.escape(term)}\b", prompt_lower):
            return chart_type
    return None


def _requested_unsupported_chart_type(prompt: str) -> str | None:
    lower = prompt.lower()
    for term in sorted(UNSUPPORTED_CHART_TERMS):
        if re.search(rf"\b{re.escape(term)}\b", lower) and term not in CHART_TYPES:
            return term
    return None


def _is_underspecified_analytics_request(prompt: str) -> bool:
    lower = prompt.lower()
    if not any(term in lower for term in ("chart", "plot", "visual", "visualize", "graph")):
        return False
    scope_terms = (
        "average",
        "avg",
        "breakdown",
        "count",
        "counts",
        "countries",
        "country",
        "date",
        "distribution",
        "language",
        "languages",
        "mean",
        "percentage",
        "percent",
        "proportion",
        "rating",
        "ratings",
        "semantic",
        "sentiment",
        "share",
        "topic",
        "topics",
        "trend",
        "verified",
    )
    return not any(term in lower for term in scope_terms)


def _mentions_rating_distribution(prompt_lower: str) -> bool:
    return bool(
        re.search(r"\bratings?\b", prompt_lower)
        and any(word in prompt_lower for word in ("distribution", "breakdown", "histogram", "counts", "count"))
    )


def _mentions_time_trend(prompt_lower: str) -> bool:
    return bool(
        any(word in prompt_lower for word in ("trend", "over time", "timeline", "time series", "by month", "monthly"))
        or (re.search(r"\bdate\b", prompt_lower) and any(word in prompt_lower for word in ("average", "avg", "by", "trend")))
    )


def _mentions_share_request(prompt_lower: str) -> bool:
    return any(word in prompt_lower for word in ("share", "percentage", "percent", "proportion", "part-to-whole"))


def _mentions_country(prompt_lower: str) -> bool:
    return bool(re.search(r"\bcountr(?:y|ies)\b", prompt_lower))
