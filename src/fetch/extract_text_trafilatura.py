"""
Stage B (final step, trafilatura): fetch page text using Athena-provided
locations via direct S3 access, extracting main content with trafilatura
instead of naive HTML-to-text stripping - much better at separating real
content from navigation/boilerplate on a single page (doesn't require
multiple pages per site to work, unlike frequency-based dedup approaches).

Resumable: skips domains already present in the output file.
"""

import json
import gzip
import io
import csv
import sys
import threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore.config import Config
import trafilatura

AWS_REGION = "us-east-1"
S3_BUCKET = "commoncrawl"

MAX_WORKERS = 10
PROGRESS_INTERVAL = 500

write_lock = threading.Lock()
s3 = boto3.client("s3", region_name=AWS_REGION, config=Config(max_pool_connections=MAX_WORKERS + 5))


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def fetch_warc_record(filename, offset, length):
    byte_range = f"bytes={offset}-{int(offset) + int(length) - 1}"
    try:
        resp = s3.get_object(Bucket=S3_BUCKET, Key=filename, Range=byte_range)
        raw_gz = resp["Body"].read()
        with gzip.GzipFile(fileobj=io.BytesIO(raw_gz)) as gz:
            return gz.read()
    except Exception:
        return None


def parse_warc_record(raw_bytes):
    """Split WARC record into WARC headers / HTTP headers / HTML body. Returns HTML str or None."""
    try:
        _warc_headers, rest = raw_bytes.split(b"\r\n\r\n", 1)
        _http_headers, body = rest.split(b"\r\n\r\n", 1)
        return body.decode("utf-8", errors="replace")
    except Exception:
        return None


def extract_main_text(html):
    """Use trafilatura to extract main content, separating it from nav/boilerplate."""
    try:
        text = trafilatura.extract(html, include_comments=False, include_tables=True,
                                    no_fallback=False)
        return text
    except Exception:
        return None


def process_row(row):
    domain = row["domain"]
    url = row["url"]
    filename = row["warc_filename"]
    offset = row["warc_record_offset"]
    length = row["warc_record_length"]

    raw = fetch_warc_record(filename, offset, length)
    if raw is None:
        return {"domain": domain, "url": url, "status": "fetch_error"}

    html = parse_warc_record(raw)
    if html is None:
        return {"domain": domain, "url": url, "status": "parse_error"}

    text = extract_main_text(html)
    if not text or len(text.strip()) == 0:
        # trafilatura found nothing extractable - very short/unusual pages sometimes
        # fail here; log separately so we can inspect rather than silently lose them
        return {"domain": domain, "url": url, "status": "empty_extraction"}

    return {
        "domain": domain, "url": url, "status": "found", "text": text,
        "warc_filename": filename, "warc_date": row.get("fetch_time"),
    }


def main(input_csv, output_jsonl, skiplog_path):
    output_path = Path(output_jsonl)
    skiplog_path = Path(skiplog_path)

    already_done = set()
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                try:
                    already_done.add(json.loads(line)["domain"])
                except Exception:
                    continue
    print(f"[{now()}] Already have {len(already_done):,} domains in output - will skip those.")

    rows_to_process = []
    with open(input_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["domain"] not in already_done:
                rows_to_process.append(row)

    total = len(rows_to_process)
    print(f"[{now()}] Processing {total:,} domains via S3 + trafilatura, {MAX_WORKERS} workers...")

    found = 0
    errors = 0
    empty = 0
    completed = 0

    out_mode = "a" if output_path.exists() else "w"
    log_mode = "a" if skiplog_path.exists() else "w"

    with open(output_path, out_mode, encoding="utf-8") as outfile, \
         open(skiplog_path, log_mode, encoding="utf-8") as skiplog, \
         ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        futures = {executor.submit(process_row, row): row for row in rows_to_process}

        for future in as_completed(futures):
            result = future.result()
            completed += 1

            with write_lock:
                if result["status"] == "found":
                    outfile.write(json.dumps(result) + "\n")
                    outfile.flush()
                    found += 1
                else:
                    skiplog.write(f"{result['status'].upper()}\tdomain={result['domain']}\turl={result['url']}\n")
                    if result["status"] == "empty_extraction":
                        empty += 1
                    else:
                        errors += 1

                if completed % PROGRESS_INTERVAL == 0:
                    print(f"[{now()}] Processed {completed:,}/{total:,} | found={found:,} "
                          f"empty={empty:,} errors={errors:,}")

    print(f"\n[{now()}] --- Done ---")
    print(f"Found:  {found:,}")
    print(f"Empty extraction: {empty:,}")
    print(f"Errors: {errors:,}")


if __name__ == "__main__":
    input_csv = sys.argv[1] if len(sys.argv) > 1 else "data/interim/athena_lookup_results_all_dedup.csv"
    output_jsonl = sys.argv[2] if len(sys.argv) > 2 else "data/interim/fetched_pages.jsonl"
    skiplog_path = sys.argv[3] if len(sys.argv) > 3 else "data/interim/stage_b_skipped_trafilatura.log"
    main(input_csv, output_jsonl, skiplog_path)
