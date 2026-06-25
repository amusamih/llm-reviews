from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (SRC_ROOT, PROJECT_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from evaluation.metrics import bootstrap_intervals, compute_multilabel_report
from evaluation.oats_label_mapping import load_oats_label_truth, validate_oats_label_dimension


DEFAULT_DATABASE = PROJECT_ROOT / "data" / "processed" / "evaluation_foundation.db"
DEFAULT_ANNOTATIONS_TABLE = "oats_amazon_finefood_opinions"


PREDICTION_SCHEMA = {
    "format": "JSON list, JSONL rows, or JSON object keyed by review_id",
    "required_fields_for_rows": ["review_id", "labels"],
    "labels_type": "list of label strings comparable to the selected OATS label dimension",
    "note": "Do not create this file from fabricated labels. Use only system predictions from an approved run.",
}


def evaluate_oats_predictions(
    *,
    database_path: str | Path = DEFAULT_DATABASE,
    annotations_table: str = DEFAULT_ANNOTATIONS_TABLE,
    label_dimension: str = "polarity",
    predictions_path: str | Path | None = None,
    bootstrap: int = 200,
    seed: int = 20260624,
) -> dict[str, Any]:
    mapping = validate_oats_label_dimension(label_dimension)
    truth = load_oats_label_truth(
        database_path,
        annotations_table,
        label_dimension=label_dimension,
    )
    labels = sorted({label for values in truth.values() for label in values})
    base = {
        "dataset_name": "OATS-ABSA Amazon FineFood",
        "database_path": str(database_path),
        "annotations_table": annotations_table,
        "label_dimension": label_dimension,
        "source_column": mapping["source_column"],
        "truth_review_count": len(truth),
        "truth_label_count": len(labels),
        "truth_labels": labels,
        "prediction_schema": PREDICTION_SCHEMA,
        "live_api_calls": False,
    }
    if predictions_path is None:
        return {
            **base,
            "prediction_generation_status": "pending_live_model_run_approval",
            "metrics_available": False,
            "metrics": None,
            "bootstrap": None,
            "limitations": [
                "No system prediction file was supplied.",
                "This is evaluation readiness only, not proposed-system performance evidence.",
            ],
        }

    predictions = load_prediction_labels(predictions_path)
    report = compute_multilabel_report(truth, predictions, labels=labels)
    intervals = bootstrap_intervals(
        truth,
        predictions,
        labels=labels,
        n_resamples=bootstrap,
        seed=seed,
    )
    return {
        **base,
        "prediction_generation_status": "predictions_loaded",
        "predictions_path": str(predictions_path),
        "prediction_review_count": len(predictions),
        "metrics_available": True,
        "metrics": report,
        "bootstrap": intervals,
        "limitations": [
            "Metrics are valid only if the prediction file comes from an approved system run.",
            "OATS labels evaluate the selected public label dimension only.",
        ],
    }


def load_prediction_labels(path: str | Path) -> dict[str, set[str]]:
    prediction_path = Path(path)
    if prediction_path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return _rows_to_predictions(rows)
    raw = json.loads(prediction_path.read_text(encoding="utf-8"))
    if isinstance(raw, Mapping):
        if "predictions" in raw and isinstance(raw["predictions"], list):
            return _rows_to_predictions(raw["predictions"])
        return {str(key): _label_set(value) for key, value in raw.items()}
    if isinstance(raw, list):
        return _rows_to_predictions(raw)
    raise ValueError("Unsupported prediction file shape")


def write_oats_eval_report(output_path: str | Path, **kwargs: Any) -> Path:
    report = evaluate_oats_predictions(**kwargs)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _rows_to_predictions(rows: list[Any]) -> dict[str, set[str]]:
    predictions: dict[str, set[str]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Prediction rows must be JSON objects")
        if "review_id" not in row:
            raise ValueError("Prediction row is missing review_id")
        predictions[str(row["review_id"])] = _label_set(row.get("labels", ()))
    return predictions


def _label_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {part.strip() for part in value.split(",") if part.strip()}
    return {str(part).strip() for part in value if str(part).strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate approved system predictions against OATS labels, or write readiness schema.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--annotations-table", default=DEFAULT_ANNOTATIONS_TABLE)
    parser.add_argument("--label-dimension", default="polarity")
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "oats" / "oats_eval_readiness_20260624.json")
    args = parser.parse_args()
    path = write_oats_eval_report(
        args.output,
        database_path=args.database,
        annotations_table=args.annotations_table,
        label_dimension=args.label_dimension,
        predictions_path=args.predictions,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    print(f"eval_report={path}")


if __name__ == "__main__":
    main()
