# An LLM-Based Multi-Agent System for Multilingual Product Review Analysis

## Overview

This repository contains a Python implementation of a multi-agent LLM system for product-review analysis. It combines SQLite-backed review data, multilingual prompt handling, route selection, semantic retrieval, structured analytics, chart generation, and optional live LLM providers behind safe offline defaults.

The default configuration uses mock/offline behavior. Live API calls are disabled unless explicitly enabled in a local `.env` file for a specific run.

The implemented workflow supports three query-time routes. Direct factual prompts are handled through validated SQL, interpretive prompts are handled through retrieval-grounded semantic reasoning, and analytics prompts are handled through constrained chart specifications rendered with predefined Matplotlib routines.

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

Generated outputs, vectorstores, local databases, raw datasets, private notes, and local environment files are excluded by `.gitignore`. Fixed evaluation prompt sets that are safe to share are tracked under `evaluation/`, including `alternative_workflow_prompts.json` and `interface_robustness_prompts.json`.

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

The `paper` extra installs LangChain, LangChain OpenAI, and FAISS dependencies used by the provider-backed LLM and semantic retrieval paths. The default offline tests do not require API keys.
The `model-substitution` extra name is retained for backward compatibility and installs provider dependencies used by the cross-model workflow evaluation.

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

## Workflow Notes

Review records are stored in product- or service-specific SQLite tables. The retrieval/load path cleans and normalizes incoming records, removes exact duplicate rows where supported, inserts the records into the matched table, and then runs enrichment in sequence: language detection and translation, topic assignment, and semantic tagging.

When FAISS semantic retrieval is enabled, the semantic reasoning path builds review documents from approved table rows and approved columns. It can reuse a matching cached FAISS index and document metadata when the table content, embedding configuration, and chunking parameters are unchanged. Cache loading avoids unsafe pickle deserialization and does not require LangChain dangerous deserialization.

## Running Offline Tests

Run the complete offline test suite:

```bash
python -m pytest
```

Targeted workflow and safeguard checks:

```bash
python -m pytest tests/test_orchestrator_query_time_flow.py tests/test_retrieval_enrichment_pipeline.py tests/test_reviewer_response_safeguards.py tests/test_semantic_tagging_and_reasoning.py
```

Targeted SQL, analytics, and implementation-alignment checks:

```bash
python -m pytest tests/test_analytics_security.py tests/test_sql_validator.py tests/test_framework_alignment.py
```

Targeted benchmark and model-interface checks:

```bash
python -m pytest tests/test_benchmark_runner.py tests/test_live_baseline_ablation.py tests/test_model_interface_robustness.py tests/test_model_substitution.py
```

## Optional Live Model Configuration

Live calls remain disabled by default. To run a configured live experiment, provide provider keys and endpoint URLs locally in `.env`, then enable the required live flags for that run.

Configured provider paths include:

- GPT-4o through the OpenAI/LangChain path.
- Claude Sonnet through the Anthropic path.
- Llama-3.3-70B-Instruct-compatible GGUF endpoint through Hugging Face Inference Endpoints.
- Qwen2.5-72B-Instruct endpoint through Hugging Face Inference Endpoints.

The manuscript-aligned cross-model workflow evaluation uses GPT-4o, Claude Sonnet, Llama-3.3-70B-Instruct, and Qwen2.5-72B-Instruct over the fixed 30-prompt set. No-call readiness checks can be run without sending benchmark prompts:

```bash
python evaluation/model_interface_robustness.py --config evaluation/model_configs.json --prompts evaluation/interface_robustness_prompts.json --preflight
```

The default preflight checks the four manuscript-aligned model configurations and may report that configured live runs are not ready until local `.env` values, endpoint URLs, and local benchmark prompt/review artifacts are provided. To run a smaller no-call availability check, pass an explicit subset:

```bash
python evaluation/model_interface_robustness.py --config evaluation/model_configs.json --prompts evaluation/interface_robustness_prompts.json --preflight --models gpt4o claude
```

## Evaluation Scripts

The benchmark runner supports offline/mock validation:

```bash
python -m evaluation.run_benchmark --mode mock
```

The tracked workflow-comparison prompt set is available at:

```text
evaluation/alternative_workflow_prompts.json
```

The tracked cross-model workflow prompt set is available at:

```text
evaluation/interface_robustness_prompts.json
```

Dataset preparation utilities are available under `scripts/` and `evaluation/`. Raw downloaded datasets, generated benchmark prompt/review artifacts, and generated SQLite databases are kept out of version control by default.

The public repository does not redistribute third-party platform reviews or the canonical local case-study database used for the manuscript's controlled quantitative results. Exact manuscript quantitative reproduction requires the corresponding local case-study data/artifacts or regenerated local data consistent with the documented schema. The code supports inspection, re-execution, and adaptation, but exact LLM-generated outputs may vary with model availability, version updates, API behavior, and runtime conditions.

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
- Unsupported or ambiguous route and analytics requests return controlled explanatory responses.
- Generated outputs, local databases, vectorstores, and raw datasets should not be committed.

## Limitations

- Offline tests use mock/local fixtures and do not imply live model performance.
- Live API behavior depends on provider availability, model versioning, endpoint configuration, and available API quota.
- Bounded benchmark results should not be interpreted as a comprehensive model leaderboard.
- Human-subject evaluation, independent translation-quality assessment, and inter-annotator agreement are outside the scope of this code repository.
