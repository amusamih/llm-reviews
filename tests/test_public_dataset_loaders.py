from pathlib import Path
import sqlite3

from llm_review_analysis.datasets import (
    APPROVED_DATASETS,
    amazon_review_to_row,
    ensure_oats_annotation_table,
    insert_oats_annotation_rows,
    iter_amazon_reviews,
    iter_oats_xml_reviews,
    oats_annotation_rows,
    oats_review_to_row,
)
from llm_review_analysis.db.schema import ensure_review_table, insert_review_rows


def test_approved_dataset_registry_covers_evaluation_needs():
    assert "amazon_reviews_2023" in APPROVED_DATASETS
    assert "oats_absa" in APPROVED_DATASETS
    assert "dataset-scale" in APPROVED_DATASETS["amazon_reviews_2023"].evaluation_tags
    assert "annotation-quality" in APPROVED_DATASETS["oats_absa"].evaluation_tags


def test_amazon_reviews_2023_sample_maps_to_review_schema():
    records = list(iter_amazon_reviews("tests/fixtures/amazon_reviews_2023_sample.jsonl"))
    rows = [amazon_review_to_row(record) for record in records]
    assert len(rows) == 2
    assert rows[0]["asin"] == "B000TEST01"
    assert rows[0]["rating"] == "5.0"
    assert rows[0]["date"] == "2024-01-01"
    assert rows[1]["content"] == "The charger stopped working after two weeks."


def test_oats_absa_xml_sample_preserves_annotations():
    reviews = list(iter_oats_xml_reviews("tests/fixtures/oats_absa_sample.xml"))
    assert len(reviews) == 2
    assert reviews[0].opinions[0].category == "BATTERY#QUALITY"
    assert reviews[0].opinions[1].polarity == "negative"
    row = oats_review_to_row(reviews[0])
    assert row["topic"] == "BATTERY#QUALITY, DELIVERY#SPEED"
    assert row["semantic_tags"] == "negative, positive"
    annotation_rows = oats_annotation_rows(reviews[0])
    assert annotation_rows[0]["entity"] == "BATTERY"
    assert annotation_rows[0]["attribute"] == "QUALITY"
    assert annotation_rows[1]["polarity"] == "negative"


def test_public_dataset_rows_insert_into_sqlite_schema(settings):
    conn = sqlite3.connect(settings.database_path)
    conn.row_factory = sqlite3.Row
    ensure_review_table(conn, "public_sample")
    rows = [
        amazon_review_to_row(record)
        for record in iter_amazon_reviews(Path("tests/fixtures/amazon_reviews_2023_sample.jsonl"))
    ]
    assert insert_review_rows(conn, "public_sample", rows) == 2
    count = conn.execute("SELECT COUNT(*) AS n FROM public_sample").fetchone()["n"]
    assert count == 2


def test_oats_opinion_annotations_insert_into_sqlite(settings):
    reviews = list(iter_oats_xml_reviews("tests/fixtures/oats_absa_sample.xml"))
    conn = sqlite3.connect(settings.database_path)
    conn.row_factory = sqlite3.Row
    ensure_oats_annotation_table(conn, "oats_annotations")
    assert insert_oats_annotation_rows(conn, "oats_annotations", reviews) == 3
    row = conn.execute(
        "SELECT category, entity, attribute, polarity FROM oats_annotations WHERE polarity = 'negative'"
    ).fetchone()
    assert row["category"] == "DELIVERY#SPEED"
    assert row["entity"] == "DELIVERY"
    assert row["attribute"] == "SPEED"

