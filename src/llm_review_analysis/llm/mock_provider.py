from __future__ import annotations

import json

from .base import LLMResponse


class MockLLMProvider:
    """Deterministic provider for tests and dry runs."""

    def __init__(self, model: str = "mock-llm") -> None:
        self.model = model

    def generate(self, prompt: str, *, purpose: str = "general", response_format: str | None = None) -> LLMResponse:
        lower = _mock_user_request(prompt).lower()
        if purpose == "route":
            if any(phrase in lower for phrase in ("how many", "count", "number of", "average", "avg")):
                content = "DIRECT_SQL"
            elif any(word in lower for word in ("plot", "chart", "visual", "distribution", "trend", "show")):
                content = "ANALYTICS"
            elif any(word in lower for word in ("why", "how", "problem", "issue", "good", "bad", "por que", "bueno", "malo", "contradictory", "mixed", "tell me about", "what about", "warranty", "translation quality")):
                content = "SEMANTICS"
            else:
                content = "DIRECT_SQL"
            return LLMResponse(content=content, model=self.model)

        if purpose == "metadata":
            return LLMResponse(
                content=json.dumps({"product_name": "sample_product", "date_range": None}),
                model=self.model,
            )

        if purpose == "chart_spec":
            if "country" in lower:
                content = {
                    "chart_type": "pie",
                    "x_field": "country",
                    "y_field": None,
                    "aggregation": "count",
                    "group_by": "country",
                    "title": "Reviews by country",
                    "x_label": "Country",
                    "y_label": "Count",
                    "language": "en",
                }
            elif "trend" in lower or "date" in lower:
                content = {
                    "chart_type": "line",
                    "x_field": "date",
                    "y_field": "rating",
                    "aggregation": "avg",
                    "group_by": "date",
                    "title": "Average rating over time",
                    "x_label": "Date",
                    "y_label": "Average rating",
                    "language": "en",
                }
            else:
                content = {
                    "chart_type": "bar",
                    "x_field": "rating",
                    "y_field": None,
                    "aggregation": "count",
                    "group_by": "rating",
                    "title": "Review rating distribution",
                    "x_label": "Rating",
                    "y_label": "Count",
                    "language": "en",
                }
            return LLMResponse(
                content=json.dumps(content),
                model=self.model,
            )

        if purpose == "language":
            return LLMResponse(content="LANGUAGE: en", model=self.model)

        if purpose == "topic_list":
            return LLMResponse(content="battery, delivery, quality, price, usability", model=self.model)

        if purpose == "topic_assign":
            for topic in ("battery", "delivery", "quality", "price", "usability"):
                if topic in lower:
                    return LLMResponse(content=topic, model=self.model)
            return LLMResponse(content="quality", model=self.model)

        return LLMResponse(content="Mock response generated without a live API call.", model=self.model)


def _mock_user_request(prompt: str) -> str:
    marker = "User request:"
    if marker in prompt:
        return prompt.rsplit(marker, 1)[-1]
    return prompt
