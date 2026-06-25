from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluation.metrics import bootstrap_intervals, compute_multilabel_report
from evaluation.oats_metrics import build_baseline_predictions, load_oats_truth


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate OATS opinion labels with multi-label metrics.")
    parser.add_argument("--database", required=True, type=Path, help="SQLite database path.")
    parser.add_argument("--annotations-table", required=True, help="Opinion-level annotation table.")
    parser.add_argument("--label-column", required=True, choices=("polarity", "category", "entity", "attribute"))
    parser.add_argument("--baseline", required=True, choices=("oracle", "majority", "empty"))
    parser.add_argument("--top-k", type=int, default=1, help="Number of labels for majority baseline.")
    parser.add_argument("--bootstrap", type=int, default=0, help="Number of bootstrap resamples.")
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap random seed.")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON report.")
    args = parser.parse_args()

    truth = load_oats_truth(args.database, args.annotations_table, label_column=args.label_column)
    predictions = build_baseline_predictions(truth, baseline=args.baseline, top_k=args.top_k)
    labels = sorted({label for label_set in truth.values() for label in label_set})
    report = {
        "database": str(args.database),
        "annotations_table": args.annotations_table,
        "label_column": args.label_column,
        "baseline": args.baseline,
        "top_k": args.top_k,
        "truth_review_count": len(truth),
        "metrics": compute_multilabel_report(truth, predictions, labels=labels),
    }
    if args.bootstrap:
        report["bootstrap"] = bootstrap_intervals(
            truth,
            predictions,
            labels=labels,
            n_resamples=args.bootstrap,
            seed=args.seed,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"Wrote {args.label_column} {args.baseline} report: "
        f"micro-F1={report['metrics']['micro']['f1']} macro-F1={report['metrics']['macro']['f1']} -> {args.output}"
    )


if __name__ == "__main__":
    main()
