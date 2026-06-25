"""Project configuration and path handling.

The implementation defaults to mock/offline operation. Live LLM calls and live
retrieval must be explicitly enabled through environment variables.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_int(value: str | None, default: int) -> int:
    if value is None or value.strip() == "":
        return default
    return int(value)


def _as_float(value: str | None, default: float) -> float:
    if value is None or value.strip() == "":
        return default
    return float(value)


@dataclass(frozen=True)
class Settings:
    project_root: Path
    database_path: Path
    output_dir: Path
    vectorstore_dir: Path
    llm_provider: str
    llm_model: str
    embedding_model: str
    semantic_retrieval_backend: str
    allow_live_llm: bool
    allow_live_retrieval: bool
    log_level: str
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2


def load_settings(env: dict[str, str] | None = None) -> Settings:
    values = env if env is not None else os.environ
    project_root = Path(values.get("LLM_REVIEW_PROJECT_ROOT", PROJECT_ROOT)).resolve()
    database_path = Path(values.get("REVIEWS_DB_PATH", project_root / "data" / "reviews.db")).resolve()
    output_dir = Path(values.get("OUTPUT_DIR", project_root / "outputs")).resolve()
    vectorstore_dir = Path(values.get("VECTORSTORE_DIR", project_root / "data" / "vectorstores")).resolve()

    llm_provider = values.get("LLM_PROVIDER", "mock").strip().lower()
    allow_live_llm = _as_bool(values.get("ALLOW_LIVE_LLM"), False)
    default_semantic_backend = "faiss" if llm_provider in {"langchain", "langchain-openai", "langchain_openai"} and allow_live_llm else "lexical"

    return Settings(
        project_root=project_root,
        database_path=database_path,
        output_dir=output_dir,
        vectorstore_dir=vectorstore_dir,
        llm_provider=llm_provider,
        llm_model=values.get("LLM_MODEL", "gpt-4o").strip(),
        llm_temperature=_as_float(values.get("LLM_TEMPERATURE"), 0.0),
        llm_max_tokens=_as_int(values.get("LLM_MAX_TOKENS"), 1024),
        llm_timeout_seconds=_as_float(values.get("LLM_TIMEOUT_SECONDS"), 60.0),
        llm_max_retries=_as_int(values.get("LLM_MAX_RETRIES"), 2),
        embedding_model=values.get("EMBEDDING_MODEL", "text-embedding-3-small").strip(),
        semantic_retrieval_backend=values.get("SEMANTIC_RETRIEVAL_BACKEND", default_semantic_backend).strip().lower(),
        allow_live_llm=allow_live_llm,
        allow_live_retrieval=_as_bool(values.get("ALLOW_LIVE_RETRIEVAL"), False),
        log_level=values.get("LOG_LEVEL", "INFO").strip().upper(),
    )


def ensure_directories(settings: Settings) -> None:
    for path in (settings.database_path.parent, settings.output_dir, settings.vectorstore_dir):
        path.mkdir(parents=True, exist_ok=True)
