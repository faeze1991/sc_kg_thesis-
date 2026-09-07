"""
Targeted repair pass: for domains ALREADY processed with the old 500-char
threshold, find those where trafilatura's text was under the NEW 800-char
threshold (meaning they'd have gotten the naive fallback under corrected
logic, but didn't). Re-fetch just those and apply naive extraction.

Everything else in fetched_pages.jsonl is left completely untouched.
"""
import json
import gzip
import io
import csv
from pathlib import Path
from datetime import datetime

import boto3
from bs4 import BeautifulSoup

NEW_THRESHOLD = 800
INPUT_JSONL = "data/interim/fetched_pages.jsonl"
LOCATIONS_CSV = "data/interim/athena_lookup_results_all_dedup.csv"
OUTPUT_JSONL = "data/interim/fetched_pages_repaired.jsonl"

s3 = boto3.client("s3", region_name="us-east-1")


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def fetch_and_naive_extract(filename, offset, length):
    byte_range = f"bytes={offset}-{int(offset) + int(length) - 1}"
    resp = s3.get_object(Bucket="commoncrawl", Key=filename, Range=byte_range)
    raw_gz = resp["Body"].read()
    with gzip.GzipFile(fileobj=io.BytesIO(raw_gz)) as gz:
        raw = gz.read()
    _warc, rest = raw.split(b"\r\n\r\n", 1)
    _http, html = rest.split(b"\r\n\r\n", 1)
    html_str = html.decode("utf-8", errors="replace")
    soup = BeautifulSoup(html_str, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)


def main():
    # Step 1: identify which domains need repair
    to_repair = set()
    total = 0
    with open(INPUT_JSONL, encoding="utf-8") as f:
        for line in f:
            total += 1
            r = json.loads(line)
            if r.get("extraction_method") == "trafilatura" and len(r["text"]) < NEW_THRESHOLD:
                to_repair.add(r["domain"])

    print(f"[{now()}] Total records: {total:,}")
    print(f"[{now()}] Need repair (trafilatura, under {NEW_THRESHOLD} chars): {len(to_repair):,}")

    # Step 2: load location info for the domains needing repair
    locations = {}
    with open(LOCATIONS_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["domain"] in to_repair:
                locations[row["domain"]] = row

    # Step 3: rewrite the file, repairing only the flagged domains
    repaired = 0
    repair_failed = 0
    with open(INPUT_JSONL, encoding="utf-8") as fin, \
         open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            r = json.loads(line)
            if r["domain"] in to_repair and r["domain"] in locations:
                row = locations[r["domain"]]
                try:
                    naive_text = fetch_and_naive_extract(
                        row["warc_filename"], row["warc_record_offset"], row["warc_record_length"])
                    if naive_text and len(naive_text.strip()) > 0:
                        r["text"] = naive_text
                        r["extraction_method"] = "naive_fallback_repaired"
                        repaired += 1
                except Exception as e:
                    repair_failed += 1
                    print(f"[{now()}] Failed to repair {r['domain']}: {e}")
            fout.write(json.dumps(r) + "\n")

            if (i + 1) % 10000 == 0:
                print(f"[{now()}] Processed {i+1:,}/{total:,} | repaired so far: {repaired:,}")

    print(f"\n[{now()}] --- Repair complete ---")
    print(f"Repaired: {repaired:,}")
    print(f"Failed:   {repair_failed:,}")
    print(f"Output written to: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
