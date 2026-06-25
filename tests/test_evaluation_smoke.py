import sqlite3

from evaluation.dataset_stats import compute_annotation_stats, compute_sqlite_dataset_stats
from llm_review_analysis.datasets import (
    ensure_oats_annotation_table,
    insert_oats_annotation_rows,
    iter_oats_xml_reviews,
)


def test_dataset_stats(sample_db, settings):
    conn, table = sample_db
    conn.close()
    stats = compute_sqlite_dataset_stats(settings.database_path, table)
    assert stats["total_reviews"] == 2
    assert "rating_distribution" in stats
    assert stats["verified_distribution"]["Verified Purchase"] == 2
    assert stats["content_length_words"]["count"] == 2
    assert stats["date_range"]["min"] == "2025-07-01"
    assert stats["missing_counts"]["language"] == 2


def test_annotation_stats(settings):
    reviews = list(iter_oats_xml_reviews("tests/fixtures/oats_absa_sample.xml"))
    conn = sqlite3.connect(settings.database_path)
    ensure_oats_annotation_table(conn, "oats_annotations")
    insert_oats_annotation_rows(conn, "oats_annotations", reviews)
    conn.close()
    stats = compute_annotation_stats(settings.database_path, "oats_annotations")
    assert stats["total_annotation_rows"] == 3
    assert stats["unique_reviews"] == 2
    assert stats["polarity_distribution"]["positive"] == 1
    assert stats["entity_distribution"]["BATTERY"] == 1
