from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_review_analysis.datasets import sample_amazon_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Create seeded Amazon Reviews 2023 JSONL samples.")
    parser.add_argument("--input", required=True, type=Path, help="Input Amazon Reviews 2023 JSONL/JSONL.GZ file.")
    parser.add_argument("--output", required=True, type=Path, help="Output sampled JSONL file.")
    parser.add_argument("--sample-size", required=True, type=int, help="Number of rows to sample.")
    parser.add_argument("--seed", required=True, type=int, help="Random seed for reproducibility.")
    parser.add_argument("--stratify", default=None, help="Optional record field to stratify by, e.g. rating.")
    parser.add_argument("--mode", choices=("random", "balanced", "proportional"), default="random")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional JSON manifest output path.")
    args = parser.parse_args()

    mode = "balanced" if args.stratify and args.mode == "random" else args.mode
    metadata = sample_amazon_jsonl(
        args.input,
        args.output,
        sample_size=args.sample_size,
        seed=args.seed,
        stratify_key=args.stratify,
        mode=mode,
    )
    manifest_path = args.manifest or args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {metadata['rows_written']} rows to {args.output}")
    print(f"Wrote sampling manifest to {manifest_path}")


if __name__ == "__main__":
    main()
