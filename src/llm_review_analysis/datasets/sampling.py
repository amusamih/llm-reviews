from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import random
from typing import Any, Iterable, Mapping

from .amazon_reviews_2023 import iter_amazon_reviews


def sample_amazon_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    *,
    sample_size: int,
    seed: int,
    stratify_key: str | None = None,
    mode: str = "random",
) -> dict[str, Any]:
    if sample_size < 0:
        raise ValueError("sample_size must be non-negative")
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if stratify_key:
        records, metadata = _stratified_sample(input_path, sample_size, seed, stratify_key, mode)
    else:
        records, metadata = _reservoir_sample(input_path, sample_size, seed)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    metadata.update(
        {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "sample_size_requested": sample_size,
            "rows_written": len(records),
            "seed": seed,
            "stratify_key": stratify_key or "",
            "mode": mode if stratify_key else "random",
        }
    )
    return metadata


def _reservoir_sample(input_path: Path, sample_size: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    reservoir: list[dict[str, Any]] = []
    rows_seen = 0
    for rows_seen, record in enumerate(iter_amazon_reviews(input_path), start=1):
        if len(reservoir) < sample_size:
            reservoir.append(record)
            continue
        index = rng.randrange(rows_seen)
        if index < sample_size:
            reservoir[index] = record
    return reservoir, {"rows_seen": rows_seen, "source_counts": {}, "output_counts": {}}


def _stratified_sample(
    input_path: Path,
    sample_size: int,
    seed: int,
    stratify_key: str,
    mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    counts = Counter(_normalized_key(record.get(stratify_key)) for record in iter_amazon_reviews(input_path))
    if mode == "balanced":
        quotas = _balanced_quotas(counts, sample_size)
    elif mode == "proportional":
        quotas = _proportional_quotas(counts, sample_size)
    else:
        raise ValueError("mode must be 'balanced' or 'proportional' when stratify_key is provided")

    rng = random.Random(seed)
    reservoirs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_by_key: Counter[str] = Counter()
    rows_seen = 0
    for rows_seen, record in enumerate(iter_amazon_reviews(input_path), start=1):
        key = _normalized_key(record.get(stratify_key))
        quota = quotas.get(key, 0)
        if quota <= 0:
            continue
        seen_by_key[key] += 1
        reservoir = reservoirs[key]
        if len(reservoir) < quota:
            reservoir.append(record)
            continue
        index = rng.randrange(seen_by_key[key])
        if index < quota:
            reservoir[index] = record

    records: list[dict[str, Any]] = []
    for key in sorted(reservoirs):
        records.extend(reservoirs[key])
    output_counts = Counter(_normalized_key(record.get(stratify_key)) for record in records)
    return records, {
        "rows_seen": rows_seen,
        "source_counts": dict(sorted(counts.items())),
        "quotas": dict(sorted(quotas.items())),
        "output_counts": dict(sorted(output_counts.items())),
    }


def _balanced_quotas(counts: Mapping[str, int], sample_size: int) -> dict[str, int]:
    keys = sorted(key for key, count in counts.items() if count > 0)
    if not keys or sample_size <= 0:
        return {key: 0 for key in keys}
    quotas = {key: 0 for key in keys}
    remaining = min(sample_size, sum(counts.values()))
    while remaining > 0:
        progressed = False
        for key in keys:
            if remaining <= 0:
                break
            if quotas[key] < counts[key]:
                quotas[key] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return quotas


def _proportional_quotas(counts: Mapping[str, int], sample_size: int) -> dict[str, int]:
    total = sum(counts.values())
    if total <= 0 or sample_size <= 0:
        return {key: 0 for key in counts}
    capped_sample_size = min(sample_size, total)
    raw = {
        key: (count / total) * capped_sample_size
        for key, count in counts.items()
    }
    quotas = {key: min(counts[key], int(value)) for key, value in raw.items()}
    remaining = capped_sample_size - sum(quotas.values())
    remainders = sorted(
        raw,
        key=lambda key: (raw[key] - int(raw[key]), counts[key]),
        reverse=True,
    )
    while remaining > 0:
        progressed = False
        for key in remainders:
            if remaining <= 0:
                break
            if quotas[key] < counts[key]:
                quotas[key] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return quotas


def _normalized_key(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()
