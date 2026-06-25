import sqlite3

from evaluation.metrics import bootstrap_intervals, compute_multilabel_report
from evaluation.oats_metrics import build_baseline_predictions, load_oats_truth
from llm_review_analysis.datasets import (
    ensure_oats_annotation_table,
    insert_oats_annotation_rows,
    iter_oats_xml_reviews,
)


def test_multilabel_report_counts_per_class_and_micro_macro():
    truth = {
        "a": {"x", "y"},
        "b": {"x"},
        "c": set(),
    }
    predictions = {
        "a": {"x"},
        "b": {"z"},
        "c": {"x"},
    }
    report = compute_multilabel_report(truth, predictions, labels=["x", "y", "z"])
    assert report["sample_count"] == 3
    assert report["per_class"]["x"]["tp"] == 1
    assert report["per_class"]["x"]["fp"] == 1
    assert report["per_class"]["x"]["fn"] == 1
    assert report["per_class"]["x"]["f1"] == 0.5
    assert report["micro"]["tp"] == 1
    assert report["micro"]["fp"] == 2
    assert report["micro"]["fn"] == 2
    assert report["micro"]["f1"] == 0.333333


def test_bootstrap_intervals_are_reported_for_summary_metrics():
    truth = {"a": {"x"}, "b": {"y"}, "c": set()}
    predictions = {"a": {"x"}, "b": {"x"}, "c": set()}
    intervals = bootstrap_intervals(truth, predictions, labels=["x", "y"], n_resamples=20, seed=7)
    assert intervals["n_resamples"] == 20
    assert "micro_f1" in intervals["metrics"]
    assert intervals["metrics"]["micro_f1"]["low"] <= intervals["metrics"]["micro_f1"]["high"]


def test_oats_truth_loading_and_baselines(settings):
    reviews = list(iter_oats_xml_reviews("tests/fixtures/oats_absa_sample.xml"))
    conn = sqlite3.connect(settings.database_path)
    ensure_oats_annotation_table(conn, "oats_annotations")
    insert_oats_annotation_rows(conn, "oats_annotations", reviews)
    conn.close()

    truth = load_oats_truth(settings.database_path, "oats_annotations", label_column="polarity")
    oracle = build_baseline_predictions(truth, baseline="oracle")
    majority = build_baseline_predictions(truth, baseline="majority")
    oracle_report = compute_multilabel_report(truth, oracle)
    majority_report = compute_multilabel_report(truth, majority)

    assert oracle_report["micro"]["f1"] == 1.0
    assert oracle_report["exact_match"] == 1.0
    assert majority_report["micro"]["f1"] < 1.0
