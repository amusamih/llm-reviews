from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


def estimate_mock_scalability(total_reviews: int, per_review_seconds: float) -> dict[str, float]:
    return {
        "total_reviews": float(total_reviews),
        "estimated_total_seconds": float(total_reviews) * per_review_seconds,
        "estimated_reviews_per_second": 1.0 / per_review_seconds if per_review_seconds else 0.0,
    }


def measure_sqlite_scalability(
    database_path: str | Path,
    *,
    tables: Iterable[str] | None = None,
    repetitions: int = 5,
) -> dict[str, Any]:
    database = Path(database_path)
    with sqlite3.connect(database) as conn:
        conn.row_factory = sqlite3.Row
        selected_tables = list(tables or _list_tables(conn))
        table_reports = []
        for table in selected_tables:
            row_count = int(conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
            timings = []
            for _ in range(max(1, repetitions)):
                start = time.perf_counter()
                conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()
                timings.append((time.perf_counter() - start) * 1000.0)
            table_reports.append(
                {
                    "table": table,
                    "row_count": row_count,
                    "count_query_latency_ms": _latency_summary(timings),
                }
            )
    total_rows = sum(report["row_count"] for report in table_reports)
    return {
        "database_path": str(database),
        "database_size_bytes": database.stat().st_size if database.exists() else 0,
        "live_api_calls": False,
        "repetitions": repetitions,
        "total_rows_across_selected_tables": total_rows,
        "tables": table_reports,
        "readiness_protocol": {
            "ingestion_time": "measure around approved loader command",
            "dataset_statistics_time": "measure around evaluation.dataset_stats or OATS stats command",
            "db_size": "record database_size_bytes",
            "faiss_index_construction_time": "measure only after live/local embedding path approval",
            "query_latency": "record per prompt in benchmark result rows",
            "analytics_latency": "record chart prompt latency in benchmark result rows",
            "memory_use": "optional; requires platform-specific sampler",
            "p50_p95_p99_latency": "available in benchmark/scalability summaries",
            "failure_retry_logging": "available in benchmark result rows and failure examples",
        },
        "limitations": [
            "This is non-live SQLite readiness measurement, not a full scalability experiment.",
            "No FAISS/live embedding construction is measured here.",
            "Do not combine heterogeneous datasets in reports without stating the composition.",
        ],
    }


def write_scalability_readiness(
    output_path: str | Path,
    *,
    database_path: str | Path,
    tables: Iterable[str] | None = None,
    repetitions: int = 5,
) -> Path:
    report = measure_sqlite_scalability(database_path, tables=tables, repetitions=repetitions)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _list_tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
    return [str(row[0]) for row in rows if str(row[0]) != "sqlite_sequence"]


def _latency_summary(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    return {
        "count": len(ordered),
        "mean": round(mean(ordered), 3),
        "p50": round(_percentile(ordered, 50), 3),
        "p95": round(_percentile(ordered, 95), 3),
        "p99": round(_percentile(ordered, 99), 3),
    }


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (percentile / 100.0) * (len(sorted_values) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = rank - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


def main() -> None:
    parser = argparse.ArgumentParser(description="Write non-live scalability/cost/latency readiness measurements.")
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--tables", nargs="*", default=None)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    path = write_scalability_readiness(
        args.output,
        database_path=args.database,
        tables=args.tables,
        repetitions=args.repetitions,
    )
    print(f"scalability_readiness={path}")


if __name__ == "__main__":
    main()
