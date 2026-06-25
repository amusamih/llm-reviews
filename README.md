# A Multi-Agent LLM Framework for Multilingual Product Review Analysis

## Overview

This repository contains a Python implementation of a multi-agent LLM framework for product-review analysis. It combines SQLite-backed review data, route selection, semantic retrieval, structured analytics, chart generation, and optional live LLM providers behind safe offline defaults.

The default configuration uses mock/offline behavior. Live API calls are disabled unless explicitly enabled in a local `.env` file for a specific run.

## Repository Structure

```text
src/                     Importable package code
app/                     Optional Flask app wrapper
evaluation/              Benchmark, dataset, model-configuration, and evaluation utilities
scripts/                 Local helper scripts
tests/                   Offline tests and sanitized fixtures
data/                    Local data workspace; generated data is git-ignored
.env.example             Placeholder-only local configuration template
pyproject.toml           Package and optional dependency groups
```

Generated outputs, vectorstores, local databases, raw datasets, private notes, and local environment files are excluded by `.gitignore`.

## Setup

Create and activate a Python environment, then install the package:

```bash
python -m pip install -e ".[dev]"
```

Optional extras:

```bash
python -m pip install -e ".[paper]"
python -m pip install -e ".[model-substitution]"
```

The `paper` extra installs LangChain, LangChain OpenAI, and FAISS dependencies used by the live LLM and retrieval paths. The default offline tests do not require API keys.

## Configuration

Copy `.env.example` to `.env` locally if you need to configure optional live calls. Do not commit `.env`.

Safe defaults:

```bash
LLM_PROVIDER=mock
ALLOW_LIVE_LLM=false
ALLOW_LIVE_RETRIEVAL=false
SEMANTIC_RETRIEVAL_BACKEND=lexical
```

`.env.example` contains placeholders for:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `HF_TOKEN`
- `HUGGINGFACEHUB_API_TOKEN`
- `HF_LLAMA_ENDPOINT_URL`
- `HF_QWEN_ENDPOINT_URL`

## Running Offline Tests

Run the complete offline test suite:

```bash
python -m pytest
```

Targeted safety checks:

```bash
python -m pytest tests/test_analytics_security.py tests/test_sql_validator.py tests/test_framework_alignment.py
```

Targeted benchmark checks:

```bash
python -m pytest tests/test_benchmark_runner.py tests/test_live_baseline_ablation.py tests/test_pilot_behavior_fixes.py
```

Targeted model-configuration checks:

```bash
python -m pytest tests/test_model_substitution.py
```

## Optional Live Model Configuration

Live calls remain disabled by default. To run a configured live experiment, provide provider keys and endpoint URLs locally in `.env`, then enable the required live flags for that run.

Configured provider paths include:

- GPT-4o through the OpenAI/LangChain path.
- Claude Sonnet through the Anthropic path.
- Llama-3.3-70B-Instruct-compatible GGUF endpoint through Hugging Face Inference Endpoints.
- Qwen2.5-72B-Instruct endpoint through Hugging Face Inference Endpoints.

No-call readiness checks can be run without sending benchmark prompts:

```bash
python evaluation/live_model_substitution.py --preflight-check
```

## Evaluation Scripts

The benchmark runner supports offline/mock validation:

```bash
python -m evaluation.run_benchmark --mode mock
```

Dataset preparation utilities are available under `scripts/` and `evaluation/`. Raw downloaded datasets and generated SQLite databases are kept out of version control by default.

Evaluation scripts may write local artifacts such as:

- `manifest.json`
- `results.jsonl`
- `summary.json`
- `evidence.json`
- `model_comparison.json`
- `cost_latency.json`
- `failure_examples.json`

These files are generated under `outputs/` and should not be committed.

## Safety and Configuration Notes

- Live calls are disabled by default.
- API keys and endpoint URLs should be supplied locally through `.env`.
- `.env.example` provides placeholders only.
- Database access for generated query paths uses SELECT-only SQL validation and table/column allowlists.
- Analytics uses structured chart specifications and predefined plotting routines.
- Generated outputs, local databases, vectorstores, and raw datasets should not be committed.

## Limitations

- Offline tests use mock/local fixtures and do not imply live model performance.
- Live API behavior depends on provider availability, model versioning, endpoint configuration, and available API quota.
- Bounded benchmark results should not be interpreted as a comprehensive model leaderboard.
- Human-subject evaluation, independent translation-quality assessment, and inter-annotator agreement are outside the scope of this code repository.
