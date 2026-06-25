from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import uuid

import pytest

from llm_review_analysis.agents import RetrievalAgent
from llm_review_analysis.config import Settings
from llm_review_analysis.llm import MockLLMProvider


@pytest.fixture
def settings() -> Settings:
    root = Path("test_runtime") / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    return Settings(
        project_root=root,
        database_path=root / "reviews.db",
        output_dir=root / "outputs",
        vectorstore_dir=root / "vectorstores",
        llm_provider="mock",
        llm_model="gpt-4o",
        embedding_model="text-embedding-3-small",
        semantic_retrieval_backend="lexical",
        allow_live_llm=False,
        allow_live_retrieval=False,
        log_level="INFO",
    )


@pytest.fixture
def provider() -> MockLLMProvider:
    return MockLLMProvider()


@pytest.fixture
def sample_rows() -> list[dict[str, str]]:
    return json.loads(Path("tests/fixtures/sample_reviews.json").read_text(encoding="utf-8"))


@pytest.fixture
def sample_db(settings: Settings, sample_rows: list[dict[str, str]]) -> tuple[sqlite3.Connection, str]:
    conn = sqlite3.connect(settings.database_path)
    conn.row_factory = sqlite3.Row
    table = RetrievalAgent(settings).load_records(conn, "sample product", sample_rows)
    return conn, table
