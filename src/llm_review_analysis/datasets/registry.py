from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetInfo:
    dataset_id: str
    name: str
    role: str
    source_url: str
    evaluation_tags: tuple[str, ...]
    notes: str


APPROVED_DATASETS: dict[str, DatasetInfo] = {
    "amazon_reviews_2023": DatasetInfo(
        dataset_id="amazon_reviews_2023",
        name="Amazon Reviews 2023",
        role=(
            "Main naturally occurring product-review corpus for dataset statistics, "
            "scalability, cost/latency, and end-to-end review-table experiments."
        ),
        source_url="https://amazon-reviews-2023.github.io/",
        evaluation_tags=("dataset-scale", "benchmark", "provenance", "baseline", "scalability"),
        notes="Use category-level JSONL.GZ samples and keep raw data out of git.",
    ),
    "oats_absa": DatasetInfo(
        dataset_id="oats_absa",
        name="OATS-ABSA",
        role=(
            "Human-annotated aspect/opinion/sentiment supplement for semantic "
            "taxonomy and per-class/per-dimension evaluation."
        ),
        source_url="https://github.com/RiTUAL-UH/OATS-ABSA",
        evaluation_tags=("semantic-labels", "annotation-quality", "semantic-taxonomy", "label-distribution"),
        notes="Use as annotated evaluation data, not as the large-scale product corpus.",
    ),
}

