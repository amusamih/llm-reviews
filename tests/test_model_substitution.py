from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from evaluation.live_model_substitution import build_model_comparison
from evaluation.live_pilot import _scrubbed_provider_config
from evaluation.model_config import load_model_configs, validate_api_keys, validate_endpoint_urls
from llm_review_analysis.config import load_settings
from llm_review_analysis.llm.huggingface_provider import HuggingFaceEndpointProvider
from llm_review_analysis.providers import build_llm_provider


CONFIG_PATH = Path("evaluation/model_configs.json")
MODEL_ENV = {
    "ANTHROPIC_MODEL_ID": "claude-sonnet-test",
    "HF_MODEL_LLAMA": "lmstudio-community/Llama-3.3-70B-Instruct-GGUF",
    "HF_MODEL_LLAMA_REPORT_LABEL": "meta-llama/Llama-3.3-70B-Instruct compatible GGUF endpoint",
}


def test_model_substitution_config_resolves_supported_models():
    configs = load_model_configs(
        CONFIG_PATH,
        env=MODEL_ENV,
        require_model_ids=True,
    )

    labels = {config.model_label for config in configs}
    assert labels == {
        "gpt4o_primary",
        "claude_sonnet_4_6_configured",
        "qwen2_5_72b_instruct_endpoint",
        "llama3_3_70b_instruct_endpoint",
    }
    assert next(config for config in configs if config.model_label == "gpt4o_primary").model_id == "gpt-4o"
    assert next(config for config in configs if config.model_label == "claude_sonnet_4_6_configured").model_id == "claude-sonnet-test"
    assert next(config for config in configs if config.model_label == "qwen2_5_72b_instruct_endpoint").model_id == "Qwen/Qwen2.5-72B-Instruct"
    llama = next(config for config in configs if config.model_label == "llama3_3_70b_instruct_endpoint")
    assert llama.model_id == "lmstudio-community/Llama-3.3-70B-Instruct-GGUF"
    assert llama.report_model_id == "meta-llama/Llama-3.3-70B-Instruct compatible GGUF endpoint"
    assert llama.endpoint_url_env == "HF_LLAMA_ENDPOINT_URL"
    assert all(config.model_id != "dummy" for config in configs)
    assert next(config for config in configs if config.model_label == "qwen2_5_72b_instruct_endpoint").endpoint_url_env == "HF_QWEN_ENDPOINT_URL"


def test_claude_model_id_must_be_configured_before_live_run():
    with pytest.raises(ValueError, match="claude_sonnet_4_6_configured"):
        load_model_configs(CONFIG_PATH, env={}, require_model_ids=True)


def test_model_substitution_api_key_validation_allows_hf_alternative_token_name():
    configs = load_model_configs(CONFIG_PATH, env=MODEL_ENV)
    missing = validate_api_keys(
        configs,
        {
            "OPENAI_API_KEY": "openai-key",
            "ANTHROPIC_API_KEY": "anthropic-key",
            "HUGGINGFACEHUB_API_TOKEN": "hf-key",
        },
    )

    assert missing == {}


def test_model_substitution_endpoint_url_validation_requires_hf_endpoint_urls():
    configs = load_model_configs(CONFIG_PATH, env=MODEL_ENV)

    missing = validate_endpoint_urls(configs, {})

    assert missing == {
        "llama3_3_70b_instruct_endpoint": ["HF_LLAMA_ENDPOINT_URL"],
        "qwen2_5_72b_instruct_endpoint": ["HF_QWEN_ENDPOINT_URL"],
    }

    assert validate_endpoint_urls(
        configs,
        {
            "HF_LLAMA_ENDPOINT_URL": "https://example.invalid/llama",
            "HF_QWEN_ENDPOINT_URL": "https://example.invalid/qwen",
        },
    ) == {}


def test_provider_factory_gates_anthropic_and_huggingface_before_optional_imports():
    anthropic_settings = load_settings({"LLM_PROVIDER": "anthropic", "LLM_MODEL": "claude-sonnet-test"})
    with pytest.raises(RuntimeError, match="Live Anthropic LLM calls are disabled"):
        build_llm_provider(anthropic_settings)

    hf_settings = load_settings({"LLM_PROVIDER": "huggingface", "LLM_MODEL": "Qwen/Qwen2.5-72B-Instruct"})
    with pytest.raises(RuntimeError, match="Live Hugging Face endpoint calls are disabled"):
        build_llm_provider(hf_settings)


def test_secret_redaction_covers_model_substitution_keys():
    scrubbed = _scrubbed_provider_config(
        {
            "OPENAI_API_KEY": "sk-openai-secret",
            "ANTHROPIC_API_KEY": "anthropic-secret",
            "HF_TOKEN": "hf-secret",
            "HUGGINGFACEHUB_API_TOKEN": "hfhub-secret",
            "HF_ENDPOINT_URL": "https://endpoint-secret.example",
            "HF_LLAMA_ENDPOINT_URL": "https://llama-secret.example",
            "HF_QWEN_ENDPOINT_URL": "https://qwen-secret.example",
            "LLM_MODEL": "gpt-4o",
        }
    )

    assert scrubbed["OPENAI_API_KEY"] == "<present redacted>"
    assert scrubbed["ANTHROPIC_API_KEY"] == "<present redacted>"
    assert scrubbed["HF_TOKEN"] == "<present redacted>"
    assert scrubbed["HUGGINGFACEHUB_API_TOKEN"] == "<present redacted>"
    assert scrubbed["HF_ENDPOINT_URL"] == "<present redacted>"
    assert scrubbed["HF_LLAMA_ENDPOINT_URL"] == "<present redacted>"
    assert scrubbed["HF_QWEN_ENDPOINT_URL"] == "<present redacted>"
    assert scrubbed["LLM_MODEL"] == "gpt-4o"


def test_huggingface_endpoint_provider_uses_base_url_and_real_payload_model(monkeypatch):
    endpoint_url = "https://qwen-endpoint.example"

    class FakeInferenceClient:
        instances = []

        def __init__(self, **kwargs):
            self.init_kwargs = kwargs
            self.chat_kwargs = None
            FakeInferenceClient.instances.append(self)

        def chat_completion(self, **kwargs):
            self.chat_kwargs = kwargs
            return {
                "choices": [{"message": {"content": "DIRECT_SQL"}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
            }

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(InferenceClient=FakeInferenceClient))
    monkeypatch.setenv("HF_TOKEN", "hf-test-token")
    monkeypatch.setenv("HF_ENDPOINT_URL", endpoint_url)

    provider = HuggingFaceEndpointProvider("Qwen/Qwen2.5-72B-Instruct", allow_live=True)
    response = provider.generate("How many reviews?", purpose="route")

    client = FakeInferenceClient.instances[0]
    assert client.init_kwargs["base_url"] == endpoint_url
    assert "model" not in client.init_kwargs
    assert client.chat_kwargs["model"] == "Qwen/Qwen2.5-72B-Instruct"
    assert client.chat_kwargs["model"] != "dummy"
    assert response.content == "DIRECT_SQL"
    assert response.usage == {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}


def test_huggingface_endpoint_provider_redacts_endpoint_url_and_token_in_errors(monkeypatch):
    endpoint_url = "https://qwen-endpoint.example"
    token = "hf-test-token"

    class FailingInferenceClient:
        def __init__(self, **kwargs):
            self.init_kwargs = kwargs

        def chat_completion(self, **kwargs):
            raise RuntimeError(f"failed at {endpoint_url}/v1/chat/completions with {token}")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(InferenceClient=FailingInferenceClient))
    monkeypatch.setenv("HF_TOKEN", token)
    monkeypatch.setenv("HF_ENDPOINT_URL", endpoint_url)

    provider = HuggingFaceEndpointProvider("Qwen/Qwen2.5-72B-Instruct", allow_live=True)
    with pytest.raises(RuntimeError) as exc_info:
        provider.generate("How many reviews?", purpose="route")

    message = str(exc_info.value)
    assert endpoint_url not in message
    assert token not in message
    assert "<redacted endpoint URL>" in message
    assert "<redacted secret>" in message


def test_build_model_comparison_returns_rows_for_configured_models():
    configs = load_model_configs(
        CONFIG_PATH,
        selected_labels=["gpt4o_primary", "qwen2_5_72b_instruct_endpoint"],
        env={},
        require_model_ids=True,
    )

    rows = build_model_comparison([], configs)

    assert [row["model_label"] for row in rows] == ["gpt4o_primary", "qwen2_5_72b_instruct_endpoint"]
    assert rows[0]["provider"] == "langchain"
    assert rows[1]["provider"] == "huggingface"
    assert rows[0]["token_usage"]["usage_available"] is False
