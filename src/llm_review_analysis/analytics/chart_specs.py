from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from llm_review_analysis.db.schema import REVIEW_COLUMNS


class ChartSpecValidationError(ValueError):
    pass


CHART_TYPES = {"bar", "line", "pie"}
AGGREGATIONS = {"count", "avg", "sum"}


@dataclass(frozen=True)
class ChartSpec:
    chart_type: str
    x_field: str
    y_field: str | None = None
    aggregation: str = "count"
    group_by: str | None = None
    title: str = "Review chart"
    x_label: str = ""
    y_label: str = ""
    language: str = "en"

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ChartSpec":
        spec = cls(
            chart_type=str(raw.get("chart_type", "bar")).lower(),
            x_field=str(raw.get("x_field", "rating")),
            y_field=str(raw["y_field"]) if raw.get("y_field") else None,
            aggregation=str(raw.get("aggregation", "count")).lower(),
            group_by=str(raw["group_by"]) if raw.get("group_by") else None,
            title=str(raw.get("title", "Review chart")),
            x_label=str(raw.get("x_label", raw.get("x_field", ""))),
            y_label=str(raw.get("y_label", raw.get("aggregation", "count"))),
            language=str(raw.get("language", "en")),
        )
        spec.validate()
        return spec

    def validate(self, allowed_columns: tuple[str, ...] = REVIEW_COLUMNS) -> None:
        if self.chart_type not in CHART_TYPES:
            raise ChartSpecValidationError(f"Unsupported chart_type: {self.chart_type}")
        if self.aggregation not in AGGREGATIONS:
            raise ChartSpecValidationError(f"Unsupported aggregation: {self.aggregation}")
        for field_name, value in (("x_field", self.x_field), ("y_field", self.y_field), ("group_by", self.group_by)):
            if value and value not in allowed_columns:
                raise ChartSpecValidationError(f"Unsupported {field_name}: {value}")
        if self.aggregation in {"avg", "sum"} and not self.y_field:
            raise ChartSpecValidationError(f"Aggregation {self.aggregation} requires y_field")
