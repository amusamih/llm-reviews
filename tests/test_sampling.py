import json

from llm_review_analysis.datasets import sample_amazon_jsonl


def test_seeded_amazon_sampling_is_reproducible(settings):
    input_path = "tests/fixtures/amazon_reviews_2023_sample.jsonl"
    output_a = settings.project_root / "sample_a.jsonl"
    output_b = settings.project_root / "sample_b.jsonl"
    meta_a = sample_amazon_jsonl(input_path, output_a, sample_size=1, seed=123)
    meta_b = sample_amazon_jsonl(input_path, output_b, sample_size=1, seed=123)
    assert meta_a["rows_written"] == 1
    assert output_a.read_text(encoding="utf-8") == output_b.read_text(encoding="utf-8")


def test_balanced_amazon_sampling_records_strata(settings):
    input_path = "tests/fixtures/amazon_reviews_2023_sample.jsonl"
    output_path = settings.project_root / "balanced.jsonl"
    metadata = sample_amazon_jsonl(
        input_path,
        output_path,
        sample_size=2,
        seed=123,
        stratify_key="rating",
        mode="balanced",
    )
    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert metadata["rows_written"] == 2
    assert metadata["output_counts"] == {"2.0": 1, "5.0": 1}
    assert {record["rating"] for record in records} == {2.0, 5.0}
