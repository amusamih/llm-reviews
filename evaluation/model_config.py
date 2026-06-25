from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ModelSubstitutionConfig:
    provider: str
    model_label: str
    model_id: str
    api_key_env: str
    api_key_env_alternatives: tuple[str, ...]
    temperature: float
    max_tokens: int
    timeout_seconds: float
    max_retries: int
    role: str
    model_family: str
    supports_system_messages: bool
    prompt_adapter: str
    notes: str = ""
    model_id_env: str | None = None
    report_model_id: str | None = None
    endpoint_url_env: str | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, env: Mapping[str, str] | None = None) -> "ModelSubstitutionConfig":
        model_id_env = _optional_str(raw.get("model_id_env"))
        model_id = _resolve_model_id(str(raw.get("model_id", "")), model_id_env=model_id_env, env=env)
        return cls(
            provider=str(raw["provider"]).strip().lower(),
            model_label=str(raw["model_label"]).strip(),
            model_id=model_id,
            model_id_env=model_id_env,
            api_key_env=str(raw["api_key_env"]).strip(),
            api_key_env_alternatives=tuple(str(value).strip() for value in raw.get("api_key_env_alternatives", ())),
            temperature=float(raw.get("temperature", 0.0)),
            max_tokens=int(raw.get("max_tokens", 1024)),
            timeout_seconds=float(raw.get("timeout_seconds", 60.0)),
            max_retries=int(raw.get("max_retries", 2)),
            role=str(raw.get("role", "comparison")).strip().lower(),
            model_family=str(raw.get("model_family", "unknown")).strip(),
            supports_system_messages=bool(raw.get("supports_system_messages", True)),
            prompt_adapter=str(raw.get("prompt_adapter", "chat_messages")).strip(),
            notes=str(raw.get("notes", "")),
            report_model_id=_resolve_optional_value(raw.get("report_model_id"), env=env),
            endpoint_url_env=_optional_str(raw.get("endpoint_url_env")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model_label": self.model_label,
            "model_id": self.model_id,
            "model_id_env": self.model_id_env,
            "api_key_env": self.api_key_env,
            "api_key_env_alternatives": list(self.api_key_env_alternatives),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "role": self.role,
            "model_family": self.model_family,
            "supports_system_messages": self.supports_system_messages,
            "prompt_adapter": self.prompt_adapter,
            "notes": self.notes,
            "report_model_id": self.report_model_id,
            "endpoint_url_env": self.endpoint_url_env,
        }

    def required_key_names(self) -> tuple[str, ...]:
        return (self.api_key_env, *self.api_key_env_alternatives)

    def has_required_key(self, env: Mapping[str, str]) -> bool:
        return any(bool(env.get(key)) for key in self.required_key_names())

    def required_endpoint_url_names(self) -> tuple[str, ...]:
        if self.provider == "huggingface" and self.endpoint_url_env:
            return (self.endpoint_url_env,)
        return ()

    def has_required_endpoint_url(self, env: Mapping[str, str]) -> bool:
        names = self.required_endpoint_url_names()
        return not names or all(bool(env.get(key)) for key in names)


def load_model_configs(
    path: str | Path,
    *,
    selected_labels: Sequence[str] | None = None,
    env: Mapping[str, str] | None = None,
    require_model_ids: bool = False,
) -> list[ModelSubstitutionConfig]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Model substitution config must contain a JSON list")
    selected = {label.strip() for label in selected_labels or () if label.strip()}
    configs = [ModelSubstitutionConfig.from_mapping(item, env=env) for item in raw]
    if selected:
        configs = [config for config in configs if config.model_label in selected]
        missing = selected.difference(config.model_label for config in configs)
        if missing:
            raise ValueError(f"Unknown model label(s): {', '.join(sorted(missing))}")
    if require_model_ids:
        missing_ids = [config.model_label for config in configs if not config.model_id or config.model_id.startswith("${")]
        if missing_ids:
            raise ValueError(
                "Model ID must be configured before live execution for: "
                + ", ".join(sorted(missing_ids))
            )
    return configs


def validate_api_keys(configs: Sequence[ModelSubstitutionConfig], env: Mapping[str, str]) -> dict[str, list[str]]:
    missing: dict[str, list[str]] = {}
    for config in configs:
        if not config.has_required_key(env):
            missing[config.model_label] = list(config.required_key_names())
    return missing


def validate_endpoint_urls(configs: Sequence[ModelSubstitutionConfig], env: Mapping[str, str]) -> dict[str, list[str]]:
    missing: dict[str, list[str]] = {}
    for config in configs:
        if not config.has_required_endpoint_url(env):
            missing[config.model_label] = list(config.required_endpoint_url_names())
    return missing


def _resolve_model_id(value: str, *, model_id_env: str | None, env: Mapping[str, str] | None) -> str:
    stripped = value.strip()
    if model_id_env and stripped == f"${{{model_id_env}}}":
        env_value = (env or {}).get(model_id_env)
        return env_value.strip() if env_value else stripped
    return stripped


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _resolve_optional_value(value: Any, *, env: Mapping[str, str] | None) -> str | None:
    text = _optional_str(value)
    if text is None:
        return None
    if text.startswith("${") and text.endswith("}"):
        env_name = text[2:-1].strip()
        if env_name:
            env_value = (env or {}).get(env_name)
            return env_value.strip() if env_value else text
    return text
