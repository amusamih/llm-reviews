from __future__ import annotations

import sqlite3
import sys
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (SRC_ROOT, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from llm_review_analysis.agents import RetrievalAgent, ReviewOrchestrator
from llm_review_analysis.config import ensure_directories, load_settings
from llm_review_analysis.providers import build_llm_provider


SAMPLE_ROWS = [
    {
        "asin": "SAMPLE1",
        "seller": "Example Seller",
        "author": "A",
        "rating": "5",
        "title": "Great battery",
        "date": "2025-07-01",
        "country": "UAE",
        "verified": "Verified Purchase",
        "content": "Great product with excellent battery life.",
    },
    {
        "asin": "SAMPLE1",
        "seller": "Example Seller",
        "author": "B",
        "rating": "2",
        "title": "Poor delivery",
        "date": "2025-07-02",
        "country": "UAE",
        "verified": "Verified Purchase",
        "content": "The product was good but delivery was bad.",
    },
]


def main() -> None:
    settings = load_settings()
    runtime_root = settings.project_root / "test_runtime" / f"smoke_{uuid4().hex}"
    settings = replace(
        settings,
        database_path=runtime_root / "reviews.db",
        output_dir=runtime_root / "outputs",
        vectorstore_dir=runtime_root / "vectorstores",
    )
    ensure_directories(settings)
    provider = build_llm_provider(settings)
    with sqlite3.connect(settings.database_path) as conn:
        conn.row_factory = sqlite3.Row
        table = RetrievalAgent(settings).load_records(conn, "sample product", SAMPLE_ROWS)
        orchestrator = ReviewOrchestrator(settings, provider)
        for prompt in (
            "How many reviews for sample product?",
            "Why are users unhappy about sample product?",
            "Show the rating distribution for sample product",
        ):
            print(prompt)
            result = orchestrator.answer(conn, prompt, product_table=table)
            if isinstance(result, dict) and "base64" in result:
                result = {**result, "base64": "<omitted>"}
            print(result)


if __name__ == "__main__":
    main()
