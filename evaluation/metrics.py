from __future__ import annotations

from collections import Counter
import random
from typing import Any, Iterable, Mapping


LabelMap = Mapping[str, set[str]]


def compute_multilabel_report(
    truth: LabelMap,
    predictions: LabelMap,
    *,
    labels: Iterable[str] | None = None,
    sample_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Compute exact per-review multi-label precision/recall/F1."""

    ids = list(sample_ids) if sample_ids is not None else sorted(set(truth) | set(predictions))
    label_list = sorted(set(labels or _collect_labels(truth, predictions)))
    counts = {
        label: {"tp": 0, "fp": 0, "fn": 0, "support": 0}
        for label in label_list
    }
    exact_matches = 0
    for review_id in ids:
        actual = set(truth.get(review_id, set()))
        predicted = set(predictions.get(review_id, set()))
        if actual == predicted:
            exact_matches += 1
        for label in label_list:
            in_actual = label in actual
            in_predicted = label in predicted
            if in_actual:
                counts[label]["support"] += 1
            if in_actual and in_predicted:
                counts[label]["tp"] += 1
            elif in_predicted and not in_actual:
                counts[label]["fp"] += 1
            elif in_actual and not in_predicted:
                counts[label]["fn"] += 1

    per_class = {
        label: _metrics_from_counts(**counts[label])
        for label in label_list
    }
    micro_counts = Counter()
    for class_counts in counts.values():
        micro_counts.update(class_counts)
    micro = _metrics_from_counts(
        tp=micro_counts["tp"],
        fp=micro_counts["fp"],
        fn=micro_counts["fn"],
        support=micro_counts["support"],
    )
    macro = _macro_average(per_class)
    return {
        "sample_count": len(ids),
        "label_count": len(label_list),
        "labels": label_list,
        "exact_match": round(exact_matches / len(ids), 6) if ids else 0.0,
        "micro": micro,
        "macro": macro,
        "per_class": per_class,
    }


def bootstrap_intervals(
    truth: LabelMap,
    predictions: LabelMap,
    *,
    labels: Iterable[str] | None = None,
    n_resamples: int = 1000,
    seed: int = 0,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    ids = sorted(set(truth) | set(predictions))
    if not ids or n_resamples <= 0:
        return {"n_resamples": 0, "confidence_level": confidence_level, "metrics": {}}
    rng = random.Random(seed)
    metric_values: dict[str, list[float]] = {
        "exact_match": [],
        "micro_precision": [],
        "micro_recall": [],
        "micro_f1": [],
        "macro_precision": [],
        "macro_recall": [],
        "macro_f1": [],
    }
    label_list = sorted(set(labels or _collect_labels(truth, predictions)))
    for _ in range(n_resamples):
        sample_ids = [rng.choice(ids) for _ in ids]
        report = compute_multilabel_report(truth, predictions, labels=label_list, sample_ids=sample_ids)
        metric_values["exact_match"].append(report["exact_match"])
        for metric in ("precision", "recall", "f1"):
            metric_values[f"micro_{metric}"].append(report["micro"][metric])
            metric_values[f"macro_{metric}"].append(report["macro"][metric])
    return {
        "n_resamples": n_resamples,
        "seed": seed,
        "confidence_level": confidence_level,
        "metrics": {
            name: _percentile_interval(values, confidence_level)
            for name, values in metric_values.items()
        },
    }


def _collect_labels(*maps: LabelMap) -> set[str]:
    labels: set[str] = set()
    for label_map in maps:
        for values in label_map.values():
            labels.update(value for value in values if value)
    return labels


def _metrics_from_counts(*, tp: int, fp: int, fn: int, support: int) -> dict[str, float | int]:
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "support": support,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _macro_average(per_class: Mapping[str, Mapping[str, float | int]]) -> dict[str, float]:
    active_classes = [
        values
        for values in per_class.values()
        if int(values["support"]) > 0 or int(values["tp"]) + int(values["fp"]) + int(values["fn"]) > 0
    ]
    if not active_classes:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {
        metric: round(sum(float(values[metric]) for values in active_classes) / len(active_classes), 6)
        for metric in ("precision", "recall", "f1")
    }


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 6)


def _percentile_interval(values: list[float], confidence_level: float) -> dict[str, float]:
    if not values:
        return {"low": 0.0, "high": 0.0, "mean": 0.0}
    ordered = sorted(values)
    alpha = 1.0 - confidence_level
    low_index = int((alpha / 2.0) * (len(ordered) - 1))
    high_index = int((1.0 - alpha / 2.0) * (len(ordered) - 1))
    return {
        "low": round(ordered[low_index], 6),
        "high": round(ordered[high_index], 6),
        "mean": round(sum(ordered) / len(ordered), 6),
    }
