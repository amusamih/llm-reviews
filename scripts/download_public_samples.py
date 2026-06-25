from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import shutil
import urllib.error
import urllib.request
import zipfile


AMAZON_REVIEW_URLS = {
    "all_beauty": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/All_Beauty.jsonl.gz",
    "cell_phones": "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Cell_Phones_and_Accessories.jsonl.gz",
}

OATS_ARCHIVE_URLS = (
    "https://github.com/RiTUAL-UH/OATS-ABSA/archive/refs/heads/main.zip",
    "https://github.com/RiTUAL-MBZUAI/OATS-ABSA/archive/refs/heads/main.zip",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download small approved public dataset samples.")
    parser.add_argument("--amazon-limit", type=int, default=1000, help="Rows to keep per Amazon Reviews 2023 category.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"), help="Raw-data output root.")
    parser.add_argument("--skip-amazon", action="store_true", help="Do not stream Amazon Reviews 2023 samples.")
    parser.add_argument("--skip-oats", action="store_true", help="Do not download OATS-ABSA archive.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_amazon:
        amazon_dir = args.output_dir / "amazon_reviews_2023"
        amazon_dir.mkdir(parents=True, exist_ok=True)
        for category, url in AMAZON_REVIEW_URLS.items():
            output_path = amazon_dir / f"{category}_sample_{args.amazon_limit}.jsonl"
            rows = stream_amazon_sample(url, output_path, args.amazon_limit)
            print(f"amazon_reviews_2023/{category}: wrote {rows} rows to {output_path}")
    if not args.skip_oats:
        oats_dir = args.output_dir / "oats_absa"
        extracted = download_oats_archive(oats_dir)
        xml_count = len(list(extracted.rglob("*.xml")))
        print(f"oats_absa: extracted archive to {extracted} with {xml_count} XML files")


def stream_amazon_sample(url: str, output_path: Path, limit: int) -> int:
    if limit <= 0:
        output_path.write_text("", encoding="utf-8")
        return 0
    count = 0
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        with gzip.GzipFile(fileobj=response) as gzipped:
            with output_path.open("w", encoding="utf-8") as output:
                for raw_line in gzipped:
                    if count >= limit:
                        break
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue
                    json.loads(line)
                    output.write(line + "\n")
                    count += 1
    return count


def download_oats_archive(output_dir: Path) -> Path:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / "oats_absa.zip"
    last_error: Exception | None = None
    for url in OATS_ARCHIVE_URLS:
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                archive_path.write_bytes(response.read())
            break
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last_error = exc
    else:
        raise RuntimeError("Could not download OATS-ABSA archive from known URLs") from last_error

    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(output_dir)
    archive_path.unlink()
    extracted_roots = [path for path in output_dir.iterdir() if path.is_dir()]
    return extracted_roots[0] if len(extracted_roots) == 1 else output_dir


if __name__ == "__main__":
    main()
