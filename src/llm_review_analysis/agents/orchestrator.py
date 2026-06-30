from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from typing import Any

from llm_review_analysis.agents.analytics_agent import AnalyticsAgent
from llm_review_analysis.agents.language_agent import LanguageAgent
from llm_review_analysis.agents.retrieval_agent import RetrievalAgent, RetrievalError
from llm_review_analysis.agents.semantic_reasoning_agent import SemanticReasoningAgent
from llm_review_analysis.config import Settings
from llm_review_analysis.db.schema import REVIEW_COLUMNS, list_review_tables, normalize_table_name, validate_identifier
from llm_review_analysis.db.sql_validator import SQLValidationError, execute_validated_select, validate_select_sql
from llm_review_analysis.llm import LLMProvider


@dataclass(frozen=True)
class PromptMetadata:
    product_name: str | None
    date_range: str | None = None


@dataclass(frozen=True)
class PromptLanguageInfo:
    original_prompt: str
    original_language: str
    internal_prompt: str


@dataclass(frozen=True)
class DirectSQLTrace:
    sql: str
    columns: tuple[str, ...]
    row_count: int
    planner: str = "deterministic"
    raw_plan: str = ""


@dataclass(frozen=True)
class ControlledFailure:
    category: str
    message: str
    reason: str


class ReviewOrchestrator:
    def __init__(
        self,
        settings: Settings,
        provider: LLMProvider,
        *,
        language_agent: LanguageAgent | None = None,
        retrieval_agent: RetrievalAgent | None = None,
        semantic_reasoning_agent: SemanticReasoningAgent | None = None,
        analytics_agent: AnalyticsAgent | None = None,
    ) -> None:
        self.settings = settings
        self.provider = provider
        self.language_agent = language_agent or LanguageAgent(provider)
        self.retrieval_agent = retrieval_agent or RetrievalAgent(settings, provider=provider)
        self.semantic_reasoning_agent = semantic_reasoning_agent or SemanticReasoningAgent(
            settings=settings,
            provider=provider,
            backend=settings.semantic_retrieval_backend,
        )
        self.analytics_agent = analytics_agent or AnalyticsAgent(settings, provider)

    def extract_metadata(self, prompt: str) -> PromptMetadata:
        date_range = _extract_date_range(prompt)
        product = None
        for pattern in (
            r"\bfor\s+([A-Za-z0-9 _-]+?)(?=\s+(?:from|between|on|in|before|after|by)\b|[?.!,]|$)",
            r"\bof\s+([A-Za-z0-9 _-]+?)(?=\s+(?:from|between|on|in|before|after|by)\b|[?.!,]|$)",
            r"\babout\s+([A-Za-z0-9 _-]+?)(?=\s+(?:from|between|on|in|before|after|by)\b|[?.!,]|$)",
        ):
            match = re.search(pattern, prompt, flags=re.IGNORECASE)
            if match:
                product = _clean_product_name(match.group(1))
                break
        return PromptMetadata(product_name=product, date_range=date_range)

    def route(self, prompt: str) -> str:
        response = self.provider.generate(_route_prompt(prompt), purpose="route").content.strip()
        decision = _extract_route_decision(response)
        if decision:
            return decision
        lower = prompt.lower()
        if any(phrase in lower for phrase in ("how many", "count", "number of", "average", "avg")):
            return "DIRECT_SQL"
        if any(word in lower for word in ("chart", "plot", "show", "visual", "trend")):
            return "ANALYTICS"
        if any(word in lower for word in ("why", "how", "issue", "problem", "good", "bad")):
            return "SEMANTICS"
        return "UNSUPPORTED"

    def answer(self, conn: sqlite3.Connection, prompt: str, *, product_table: str | None = None):
        result, _ = self.answer_with_trace(conn, prompt, product_table=product_table)
        return result

    def answer_with_trace(self, conn: sqlite3.Connection, prompt: str, *, product_table: str | None = None):
        language_info = self._prepare_prompt(prompt)
        internal_prompt = language_info.internal_prompt
        metadata = self.extract_metadata(internal_prompt)
        table = product_table or self._match_table(conn, metadata.product_name)
        retrieval_attempted = False
        retrieval_error = None
        if not table and metadata.product_name:
            retrieval_attempted = True
            table, retrieval_error = self._retrieve_missing_table(conn, metadata.product_name)
        trace_product_name = _trace_product_name(metadata.product_name, table)
        trace = {
            "prompt": prompt,
            "original_prompt": language_info.original_prompt,
            "original_language": language_info.original_language,
            "internal_prompt": internal_prompt,
            "product_name": trace_product_name,
            "date_range": metadata.date_range,
            "table": table,
            "retrieval_attempted": retrieval_attempted,
            "retrieval_error": retrieval_error,
            "route": None,
            "sql": None,
            "evidence_ids": [],
            "evidence_snippets": [],
            "chart_path": None,
            "chart_type": None,
            "failure_category": None,
            "failure_reason": None,
        }
        if not table:
            failure = (
                _retrieval_failure(metadata.product_name, retrieval_error)
                if retrieval_attempted and retrieval_error
                else _unknown_product_failure(metadata.product_name)
            )
            _apply_controlled_failure(trace, failure)
            return self._failure_result_for_language(failure, language_info.original_language), trace
        decision = self.route(internal_prompt)
        trace["route"] = decision
        if decision == "ANALYTICS":
            result = self.analytics_agent.run(conn, table, prompt)
            _copy_result_failure(result, trace)
            trace.update(
                {
                    "chart_path": result.get("path"),
                    "chart_type": result.get("chart_type"),
                    "chart_group_by": result.get("group_by"),
                    "chart_aggregation": result.get("aggregation"),
                    "chart_rows": result.get("chart_rows", []),
                }
            )
            return result, trace
        if decision == "SEMANTICS":
            controlled_failure = self._semantic_controlled_failure(conn, table, internal_prompt, metadata.product_name)
            if controlled_failure:
                _apply_controlled_failure(trace, controlled_failure)
                return self._failure_result_for_language(controlled_failure, language_info.original_language), trace
            semantic_trace = self.semantic_reasoning_agent.answer_with_trace(conn, table, internal_prompt)
            trace["evidence_ids"] = list(semantic_trace.evidence_ids)
            trace["evidence_snippets"] = list(semantic_trace.evidence_snippets)
            return {"type": "text", "message": self._translate_response_text(semantic_trace.answer, language_info.original_language)}, trace
        if decision == "DIRECT_SQL":
            message, direct_trace = self._run_direct_sql(conn, table, internal_prompt)
            trace["sql"] = direct_trace.sql
            trace["sql_columns"] = list(direct_trace.columns)
            trace["sql_row_count"] = direct_trace.row_count
            trace["sql_planner"] = direct_trace.planner
            return {"type": "text", "message": self._translate_response_text(message, language_info.original_language)}, trace

        failure = _unsupported_route_failure(decision)
        _apply_controlled_failure(trace, failure)
        return self._failure_result_for_language(failure, language_info.original_language), trace

    def _prepare_prompt(self, prompt: str) -> PromptLanguageInfo:
        language, translation = self.language_agent.detect_and_translate_text(prompt)
        internal_prompt = translation or prompt
        return PromptLanguageInfo(
            original_prompt=prompt,
            original_language=language or "en",
            internal_prompt=internal_prompt,
        )

    def _retrieve_missing_table(self, conn: sqlite3.Connection, product_name: str) -> tuple[str | None, str | None]:
        try:
            if hasattr(self.retrieval_agent, "retrieve_live"):
                raw_table = self.retrieval_agent.retrieve_live(product_name)
            elif hasattr(self.retrieval_agent, "fetch"):
                raw_table = self.retrieval_agent.fetch(product_name)
            else:
                return None, "Retrieval agent does not expose a retrieval method."
            if not raw_table:
                return None, "Retrieval did not return a table name."
            table = validate_identifier(str(raw_table))
            if table not in list_review_tables(conn):
                return None, f"Retrieved table was not found in the local database: {table}."
            return table, None
        except (RetrievalError, RuntimeError, NotImplementedError, ValueError) as exc:
            return None, str(exc)
        except Exception as exc:  # noqa: BLE001 - retrieval adapters should fail closed.
            return None, f"Retrieval failed: {exc}"

    def _translate_response_text(self, text: str, target_language: str) -> str:
        if _is_english_language(target_language):
            return text
        return self.language_agent.translate_text(text, target_language)

    def _failure_result_for_language(self, failure: ControlledFailure, target_language: str) -> dict[str, str]:
        result = _failure_result(failure)
        result["message"] = self._translate_response_text(result["message"], target_language)
        return result

    def _match_table(self, conn: sqlite3.Connection, product_name: str | None) -> str | None:
        tables = list_review_tables(conn)
        if not tables:
            return None
        if product_name:
            candidates = _product_table_candidates(product_name)
            matches = [candidate for candidate in candidates if candidate in tables]
            if len(matches) == 1:
                return matches[0]
            return None
        return tables[0] if len(tables) == 1 else None

    def display_product_name(self, product_name: str | None, table: str | None) -> str | None:
        return _trace_product_name(product_name, table)

    def _semantic_controlled_failure(
        self,
        conn: sqlite3.Connection,
        table: str,
        prompt: str,
        product_name: str | None,
    ) -> ControlledFailure | None:
        if _is_context_dependent_prompt(prompt):
            return ControlledFailure(
                "context_missing",
                "This question depends on earlier conversational context that is not available. Please restate the product, review, or comparison target.",
                "Prompt refers to prior context that is absent from the current request.",
            )
        if _is_ambiguous_prompt(prompt):
            return ControlledFailure(
                "ambiguous_prompt",
                "The request is ambiguous. Please specify whether you want a factual count, an evidence-based explanation, or a chart/analytics view.",
                "Prompt does not specify an actionable review-analysis task.",
            )
        if _requests_translation_quality(prompt):
            return ControlledFailure(
                "translation_quality_not_evaluated",
                "Translation-quality evaluation requires reference translations or bilingual human assessment, which are not available in the current review data.",
                "No reference-translation field or bilingual assessment data is present.",
            )
        missing_term = _missing_information_term(conn, table, prompt, product_name)
        if missing_term:
            return ControlledFailure(
                "missing_information",
                f"No matching review evidence was found for '{missing_term}'. I cannot infer that fact from the available data.",
                f"No available review text or semantic metadata contains the requested term: {missing_term}.",
            )
        return None

    def _answer_direct_sql(self, conn: sqlite3.Connection, table: str, prompt: str) -> str:
        message, _ = self._run_direct_sql(conn, table, prompt)
        return message

    def plan_direct_sql(self, table: str, prompt: str) -> str:
        return _deterministic_direct_sql(table, prompt)

    def _plan_direct_sql_with_trace(self, table: str, prompt: str) -> tuple[str, str, str]:
        deterministic_sql = _deterministic_direct_sql(table, prompt)
        if self.settings.llm_provider == "mock" or not self.settings.allow_live_llm:
            return deterministic_sql, "deterministic", ""

        response_content = ""
        try:
            response = self.provider.generate(
                _sql_plan_prompt(table, prompt),
                purpose="sql_generation",
                response_format="json",
            )
            response_content = response.content.strip()
            proposed_sql = _extract_sql_from_response(response_content)
            cleaned_sql = validate_select_sql(
                proposed_sql,
                allowed_tables=[table],
                allowed_columns=REVIEW_COLUMNS,
            )
            return cleaned_sql, "live_llm_validated", response_content
        except (SQLValidationError, ValueError, TypeError, json.JSONDecodeError):
            return deterministic_sql, "deterministic_fallback_after_live_sql", response_content

    def _run_direct_sql(self, conn: sqlite3.Connection, table: str, prompt: str) -> tuple[str, DirectSQLTrace]:
        sql, planner, raw_plan = self._plan_direct_sql_with_trace(table, prompt)
        columns, rows = execute_validated_select(conn, sql, allowed_tables=[table], allowed_columns=REVIEW_COLUMNS)
        if not rows:
            return "No data found.", DirectSQLTrace(sql=sql, columns=tuple(columns), row_count=0, planner=planner, raw_plan=raw_plan)
        values = dict(zip(columns, rows[0]))
        trace = DirectSQLTrace(sql=sql, columns=tuple(columns), row_count=len(rows), planner=planner, raw_plan=raw_plan)
        if "avg_rating" in values:
            value = values["avg_rating"]
            message = f"The average rating is {value:.2f}." if value is not None else "No numeric ratings were found."
            return message, trace
        return f"The table contains {values.get('review_count', 0)} reviews.", trace

    def build_langchain_tools(self, conn: sqlite3.Connection, table: str) -> list[object]:
        from llm_review_analysis.agents.langchain_tools import build_langchain_tools

        return build_langchain_tools(self.settings, self.provider, conn, table, orchestrator=self)


def _deterministic_direct_sql(table: str, prompt: str) -> str:
    lower = prompt.lower()
    where_clause = _date_where_clause(prompt)
    if "average" in lower or "avg" in lower:
        return f"SELECT AVG(CAST(rating AS REAL)) AS avg_rating FROM {table}{where_clause}"
    elif "how many" in lower or "count" in lower or "number" in lower:
        return f"SELECT COUNT(*) AS review_count FROM {table}{where_clause}"
    return f"SELECT COUNT(*) AS review_count FROM {table}{where_clause}"


def _clean_product_name(raw: str) -> str:
    cleaned = raw.strip(" ?.!,:;")
    cleaned = re.sub(r"\s+", " ", cleaned)
    for suffix in (
        " reviews",
        " review records",
        " review data",
        " delivery",
        " price",
        " quality",
        " warranty",
        " rating",
        " ratings",
        " distribution",
        " trend",
        " broken",
        " misleading",
        " low",
        " high",
    ):
        if cleaned.lower().endswith(suffix):
            cleaned = cleaned[: -len(suffix)].strip()
    return cleaned


def _product_table_candidates(product_name: str) -> list[str]:
    candidates: list[str] = []
    raw_values = [product_name, _clean_product_name(product_name)]
    for raw in raw_values:
        parts = [raw]
        lower = raw.lower()
        if lower.endswith(" reviews"):
            parts.append(raw[: -len(" reviews")])
        if lower.endswith(" review"):
            parts.append(raw[: -len(" review")])
        for value in parts:
            try:
                candidate = normalize_table_name(value)
            except ValueError:
                continue
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates


def _trace_product_name(product_name: str | None, table: str | None) -> str | None:
    if not table:
        return product_name
    if not product_name:
        return table.replace("_", " ")
    try:
        normalized_product = normalize_table_name(product_name)
    except ValueError:
        normalized_product = ""
    if normalized_product != table and table in _product_table_candidates(product_name):
        return table.replace("_", " ")
    return product_name


def _unknown_product_failure(product_name: str | None) -> ControlledFailure:
    target = product_name or "the requested product"
    return ControlledFailure(
        "product_not_found",
        f"No matching data found for {target}. The available tables contain 0 reviews for that requested product.",
        "No review table matched the requested product/table name.",
    )


def _retrieval_failure(product_name: str | None, reason: str | None) -> ControlledFailure:
    target = product_name or "the requested product"
    return ControlledFailure(
        "retrieval_failed",
        f"Review retrieval or enrichment could not be completed for {target}.",
        reason or "Retrieval did not produce a usable local review table.",
    )


def _unsupported_route_failure(route: str) -> ControlledFailure:
    return ControlledFailure(
        "unsupported_route",
        "The request could not be routed to a supported processing pathway. Please ask for a factual answer, an evidence-based explanation, or a supported chart.",
        f"Unsupported route decision: {route}.",
    )


def _failure_result(failure: ControlledFailure) -> dict[str, str]:
    return {
        "type": "text",
        "message": failure.message,
        "failure_category": failure.category,
        "failure_reason": failure.reason,
    }


def _apply_controlled_failure(trace: dict[str, Any], failure: ControlledFailure) -> None:
    trace["failure_category"] = failure.category
    trace["failure_reason"] = failure.reason


def _copy_result_failure(result: dict[str, Any], trace: dict[str, Any]) -> None:
    category = result.get("failure_category")
    if category:
        trace["failure_category"] = str(category)
        trace["failure_reason"] = str(result.get("failure_reason") or result.get("message") or "")


def _is_english_language(language: str) -> bool:
    return language.strip().lower() in {"en", "eng", "english"}


def _is_context_dependent_prompt(prompt: str) -> bool:
    lower = prompt.lower()
    return bool(re.search(r"\b(second|previous|earlier|that one|this one|the other one)\b", lower))


def _is_ambiguous_prompt(prompt: str) -> bool:
    normalized = re.sub(r"\s+", " ", prompt.strip().lower().rstrip(".?!"))
    return bool(
        re.fullmatch(r"(tell me about|what about|say something about|summarize)\s+.+", normalized)
        and not any(
            keyword in normalized
            for keyword in (
                "rating",
                "review",
                "reviews",
                "delivery",
                "price",
                "quality",
                "chart",
                "plot",
                "count",
                "average",
                "why",
                "how",
            )
        )
    )


def _requests_translation_quality(prompt: str) -> bool:
    lower = prompt.lower()
    return "translation quality" in lower or "quality of translation" in lower


SEMANTIC_STOPWORDS = {
    "about",
    "amazon",
    "available",
    "because",
    "does",
    "evaluate",
    "explain",
    "for",
    "from",
    "have",
    "mention",
    "mentions",
    "product",
    "products",
    "review",
    "reviews",
    "score",
    "that",
    "the",
    "this",
    "users",
    "what",
    "which",
    "with",
}


def _missing_information_term(
    conn: sqlite3.Connection,
    table_name: str,
    prompt: str,
    product_name: str | None,
) -> str | None:
    target_terms = _explicit_missing_info_targets(prompt)
    if not target_terms:
        return None
    haystack = _review_text_haystack(conn, table_name)
    product_terms = set(_tokens(product_name or ""))
    for term in target_terms:
        normalized_term = term.lower()
        if normalized_term in product_terms or normalized_term in SEMANTIC_STOPWORDS:
            continue
        if normalized_term not in haystack:
            return term
    return None


def _explicit_missing_info_targets(prompt: str) -> list[str]:
    lower = prompt.lower()
    targets: list[str] = []
    for match in re.finditer(r"\b([a-z][a-z-]{2,})\s+score\b", lower):
        targets.append(match.group(1))
    if not targets:
        return []
    return targets


def _review_text_haystack(conn: sqlite3.Connection, table_name: str) -> str:
    table = validate_identifier(table_name)
    columns, rows = execute_validated_select(
        conn,
        f"SELECT title, content, translated_review, semantic_tags FROM {table}",
        allowed_tables=[table],
        allowed_columns=REVIEW_COLUMNS,
    )
    return "\n".join(
        " ".join(str(value or "") for value in row)
        for row in rows
    ).lower()


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _extract_date_range(prompt: str) -> str | None:
    date_pattern = r"\d{4}-\d{2}-\d{2}"
    between_match = re.search(
        rf"\bbetween\s+({date_pattern})\s+and\s+({date_pattern})\b",
        prompt,
        flags=re.IGNORECASE,
    )
    if between_match:
        return f"{between_match.group(1)}..{between_match.group(2)}"
    from_match = re.search(
        rf"\bfrom\s+({date_pattern})\s+to\s+({date_pattern})\b",
        prompt,
        flags=re.IGNORECASE,
    )
    if from_match:
        return f"{from_match.group(1)}..{from_match.group(2)}"
    on_match = re.search(rf"\bon\s+({date_pattern})\b", prompt, flags=re.IGNORECASE)
    if on_match:
        return on_match.group(1)
    return None


def _date_where_clause(prompt: str) -> str:
    date_range = _extract_date_range(prompt)
    if not date_range:
        return ""
    if ".." in date_range:
        start_date, end_date = date_range.split("..", 1)
        return f" WHERE date >= '{start_date}' AND date <= '{end_date}'"
    return f" WHERE date = '{date_range}'"


def _route_prompt(prompt: str) -> str:
    return (
        "Classify the user request for a review-analysis multi-agent system.\n"
        "Return exactly one route label and no other text:\n"
        "- DIRECT_SQL: factual count/average/date-filter questions answerable by SQLite aggregation.\n"
        "- ANALYTICS: chart, plot, distribution, trend, or visual analytics requests.\n"
        "- SEMANTICS: explanation, evidence, ambiguity, missing-information, or qualitative reasoning requests.\n\n"
        f"User request: {prompt}\n"
        "Route:"
    )


def _extract_route_decision(response: str) -> str | None:
    upper = response.upper()
    for route in ("DIRECT_SQL", "SEMANTICS", "ANALYTICS"):
        if re.search(rf"\b{route}\b", upper):
            return route
    return None


def _sql_plan_prompt(table: str, prompt: str) -> str:
    allowed_columns = ", ".join(REVIEW_COLUMNS)
    return (
        "Generate a safe SQLite query for a review table. Return JSON only with one key named sql.\n"
        "Rules: use exactly one SELECT statement, no comments, no mutation statements, no joins, no subqueries, "
        "and only the listed table and columns. For factual count questions use COUNT(*) AS review_count. "
        "For average rating questions use AVG(CAST(rating AS REAL)) AS avg_rating. "
        "If the user gives an ISO date range, filter on the date column inclusively.\n\n"
        f"Table: {table}\n"
        f"Allowed columns: {allowed_columns}\n"
        f"User request: {prompt}"
    )


def _extract_sql_from_response(response: str) -> str:
    try:
        parsed: Any = json.loads(response)
    except json.JSONDecodeError:
        match = re.search(r"SELECT\s+.+", response, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            raise
        return match.group(0).strip().rstrip("`")
    if not isinstance(parsed, dict):
        raise ValueError("SQL plan response must be a JSON object")
    sql = parsed.get("sql")
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("SQL plan response must contain a non-empty sql string")
    return sql.strip()
