"""
Stage B (final step): fetch page text for every domain using the
filename/offset/length locations Athena already found, and extract clean
text - reusing the same fetch/parse/extract logic already validated
earlier in this project.

No rate-limit constraint here (data.commoncrawl.org is CDN-backed, not
the CDX index server), so this runs with a thread pool for speed.

Resumable: on startup, checks which domains already exist in the output
file and skips them - no separate checkpoint needed.
"""

import json
import gzip
import io
import csv
import sys
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup

DATA_ENDPOINT = "https://data.commoncrawl.org/{filename}"
USER_AGENT = "SC-KG-Thesis-Research/1.0 (Leuphana University; academic research use; contact: faezeh@leuphana.de)"
REQUEST_TIMEOUT = 20
MAX_RETRIES = 2
MAX_WORKERS = 10
PROGRESS_INTERVAL = 200

HEADERS = {"User-Agent": USER_AGENT}
write_lock = threading.Lock()


def fetch_warc_record(filename, offset, length):
    url = DATA_ENDPOINT.format(filename=filename)
    byte_range = f"bytes={offset}-{int(offset) + int(length) - 1}"
    headers = {**HEADERS, "Range": byte_range}
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            if resp.status_code in (200, 206):
                with gzip.GzipFile(fileobj=io.BytesIO(resp.content)) as gz:
                    return gz.read()
            return None
        except Exception:
            if attempt < MAX_RETRIES - 1:
                continue
            return None
    return None


def parse_warc_record(raw_bytes):
    try:
        _warc_headers, rest = raw_bytes.split(b"\r\n\r\n", 1)
        _http_headers, body = rest.split(b"\r\n\r\n", 1)
        return body.decode("utf-8", errors="replace")
    except Exception:
        return None


def html_to_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)


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

    text = html_to_text(html)
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
    print(f"Already have {len(already_done):,} domains in output file - will skip those.")

    rows_to_process = []
    with open(input_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["domain"] not in already_done:
                rows_to_process.append(row)

    total = len(rows_to_process)
    print(f"Processing {total:,} remaining domains with {MAX_WORKERS} workers...")

    found = 0
    errors = 0
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
                    errors += 1

                if completed % PROGRESS_INTERVAL == 0:
                    print(f"Processed {completed:,}/{total:,} | found={found:,} errors={errors:,}")

    print(f"\n--- Done ---")
    print(f"Found:  {found:,}")
    print(f"Errors: {errors:,}")


if __name__ == "__main__":
    input_csv = sys.argv[1] if len(sys.argv) > 1 else "data/interim/athena_lookup_results_dedup.csv"
    output_jsonl = sys.argv[2] if len(sys.argv) > 2 else "data/interim/fetched_pages.jsonl"
    skiplog_path = sys.argv[3] if len(sys.argv) > 3 else "data/interim/stage_b_skipped_final.log"
    main(input_csv, output_jsonl, skiplog_path)
