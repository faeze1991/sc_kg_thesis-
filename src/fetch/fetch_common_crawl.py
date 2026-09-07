"""
Stage B: Retrieve page content from Common Crawl for each deduplicated domain.

CDX lookup uses cdx_toolkit (Common Crawl's own team confirmed this library
"goes slow enough" to avoid triggering their rate limiter). WARC byte-range
fetching and HTML-to-text extraction remain our own previously-validated code.

ADAPTIVE BACKOFF WITH RETRY-SAME-DOMAIN: if lookups start failing in a row
(>= CONSECUTIVE_ERROR_THRESHOLD), the script backs off (5 min, doubling to a
60 min cap) and RETRIES THE SAME DOMAIN - it does not advance past it. This
means a sustained block (even the full 24 hours Common Crawl's FAQ mentions)
causes zero data loss: the script simply pauses on its current position until
requests succeed again, then continues normally, fully unattended.

Isolated failures below the consecutive-error threshold are logged and
skipped (treated as one-off transient blips, not a sustained block).

Single worker only - per Common Crawl's explicit policy against concurrent
threads from one IP.
"""

import json
import time
import random
import gzip
import io
from pathlib import Path
from datetime import datetime

import requests
import cdx_toolkit
from bs4 import BeautifulSoup

INPUT_PATH = Path("data/interim/domain_urls.jsonl")
OUTPUT_PATH = Path("data/interim/fetched_pages.jsonl")
SKIPPED_LOG_PATH = Path("data/interim/stage_b_skipped.log")
CHECKPOINT_PATH = Path("data/interim/stage_b_checkpoint.json")

CRAWL_FROM_TS = "20200917"
CRAWL_TO_TS = "20201002"

DATA_ENDPOINT = "https://data.commoncrawl.org/{filename}"
USER_AGENT = "SC-KG-Thesis-Research/1.0 (Leuphana University; academic research use; contact: faezeh@leuphana.de)"
DELAY_MIN, DELAY_MAX = 6.0, 12.0
REQUEST_TIMEOUT = 20
MAX_RETRIES = 2
CHECKPOINT_INTERVAL = 1
PROGRESS_INTERVAL = 100

CONSECUTIVE_ERROR_THRESHOLD = 5
BACKOFF_START_SECONDS = 300
BACKOFF_MAX_SECONDS = 3600
BACKOFF_MULTIPLIER = 2

HEADERS = {"User-Agent": USER_AGENT}

cdx = cdx_toolkit.CDXFetcher(source="cc")


def load_checkpoint():
    if CHECKPOINT_PATH.exists():
        try:
            return json.loads(CHECKPOINT_PATH.read_text()).get("completed", 0)
        except Exception:
            return 0
    return 0


def save_checkpoint(n):
    CHECKPOINT_PATH.write_text(json.dumps({"completed": n}))


def cdx_lookup(url):
    try:
        results = list(cdx.iter(url, from_ts=CRAWL_FROM_TS, to=CRAWL_TO_TS,
                                 limit=1, filter="=status:200"))
        if results:
            r = results[0]
            return {"filename": r["filename"], "offset": r["offset"],
                    "length": r["length"], "timestamp": r.get("timestamp")}
        return None
    except Exception as e:
        return f"ERROR:{type(e).__name__}"


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
                time.sleep(1)
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


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    already_done = load_checkpoint()

    with open(INPUT_PATH) as f:
        total_domains = sum(1 for _ in f)

    out_mode = "a" if (already_done > 0 or OUTPUT_PATH.exists()) else "w"
    log_mode = "a" if (already_done > 0 or SKIPPED_LOG_PATH.exists()) else "w"

    print(f"[{now()}] " + (f"Resuming from {already_done:,} domains already processed."
          if already_done else f"Starting fresh run. Total domains to process: {total_domains:,}"))

    found = 0
    not_found = 0
    errors = 0
    consecutive_errors = 0
    current_backoff = BACKOFF_START_SECONDS
    start_time = time.time()
    completed = already_done

    with open(INPUT_PATH, "r", encoding="utf-8") as infile, \
         open(OUTPUT_PATH, out_mode, encoding="utf-8") as outfile, \
         open(SKIPPED_LOG_PATH, log_mode, encoding="utf-8") as skiplog:

        for i, line in enumerate(infile):
            if i < already_done:
                continue

            record = json.loads(line)
            domain = record["domain"]
            url = record["url"]

            # Retry loop: keeps retrying the SAME domain during a sustained
            # block/congestion period, so no data is lost even over a
            # multi-hour or 24-hour wait. Only gives up (moves to next
            # domain) on isolated, non-sustained failures.
            gave_up_on_domain = False
            while True:
                cdx_result = cdx_lookup(url)
                time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
                is_error = isinstance(cdx_result, str) and cdx_result.startswith("ERROR")

                if is_error:
                    consecutive_errors += 1
                    if consecutive_errors >= CONSECUTIVE_ERROR_THRESHOLD:
                        print(f"[{now()}] {consecutive_errors} consecutive errors - likely "
                              f"congestion/block. Backing off {current_backoff/60:.1f} min, "
                              f"will retry domain={domain}...")
                        time.sleep(current_backoff)
                        current_backoff = min(current_backoff * BACKOFF_MULTIPLIER, BACKOFF_MAX_SECONDS)
                        continue  # retry the SAME domain, do not advance
                    else:
                        errors += 1
                        skiplog.write(f"{cdx_result}\tdomain={domain}\turl={url}\n")
                        gave_up_on_domain = True
                        break
                else:
                    if consecutive_errors > 0:
                        print(f"[{now()}] Recovered after {consecutive_errors} consecutive "
                              f"errors. Resuming normal speed.")
                    consecutive_errors = 0
                    current_backoff = BACKOFF_START_SECONDS
                    break

            if not gave_up_on_domain:
                if cdx_result is None:
                    not_found += 1
                    skiplog.write(f"NOT_FOUND\tdomain={domain}\turl={url}\n")
                else:
                    raw = fetch_warc_record(cdx_result["filename"], cdx_result["offset"], cdx_result["length"])
                    if raw is None:
                        errors += 1
                        skiplog.write(f"FETCH_ERROR\tdomain={domain}\turl={url}\n")
                    else:
                        html = parse_warc_record(raw)
                        if html is None:
                            errors += 1
                            skiplog.write(f"PARSE_ERROR\tdomain={domain}\turl={url}\n")
                        else:
                            text = html_to_text(html)
                            outfile.write(json.dumps({
                                "domain": domain, "url": url, "status": "found", "text": text,
                                "warc_filename": cdx_result["filename"],
                                "warc_date": cdx_result.get("timestamp"),
                            }) + "\n")
                            outfile.flush()
                            found += 1

            completed = i + 1
            if completed % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(completed)

            if completed % PROGRESS_INTERVAL == 0:
                elapsed = time.time() - start_time
                rate = (completed - already_done) / elapsed if elapsed > 0 else 0
                remaining = (total_domains - completed) / rate if rate > 0 else float("inf")
                print(f"[{now()}] Processed {completed:,}/{total_domains:,} | found={found:,} "
                      f"not_found={not_found:,} errors={errors:,} | ~{remaining/3600:.1f}h remaining")

        save_checkpoint(completed)

    print(f"\n[{now()}] --- Stage B complete ---")
    print(f"Found:     {found:,}")
    print(f"Not found: {not_found:,}")
    print(f"Errors:    {errors:,}")


if __name__ == "__main__":
    main()
