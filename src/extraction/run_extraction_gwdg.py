"""
Stage C production extraction script.

Design informed by:
- Our own validated prompt (relevance check + tightened format, tested on
  40 pages x 3 models with 100% JSON parse success and ~45% signal rate)
- Documented GWDG/KISSKI server-side instability from a related thesis
  project on the same infrastructure ("checkpoint/resume logic is more
  impactful than timeout tuning") - so this prioritizes robust retry +
  resumability over aggressive timeout settings

Runs ONE model at a time (pass model name as argument) - the full 3-model
comparison stays on the smaller evaluation sample; production runs use a
single chosen model for the full corpus.

Resumable: skips domains already present in the output file, same pattern
as Stage B. Checkpointed per-record (via output file itself, not a
separate checkpoint file - safe under concurrency since order doesn't matter).

Adaptive backoff: if requests start failing in a row, pauses and retries
with increasing wait time, rather than hammering an unstable server.
"""
import json
import os
import re
import sys
import time
import random
import threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

API_KEY = os.environ["GWDG_API_KEY"]
ENDPOINT = os.environ.get("EXTRACTION_ENDPOINT", "https://chat-ai.academiccloud.de/v1/chat/completions")

MAX_WORKERS = 3
REQUEST_TIMEOUT = 60     # per Shima's finding: longer timeouts don't help server hangs
MAX_RETRIES = 3
PROGRESS_INTERVAL = 100

CONSECUTIVE_ERROR_THRESHOLD = 5
BACKOFF_START_SECONDS = 30
BACKOFF_MAX_SECONDS = 600
BACKOFF_MULTIPLIER = 2

PROMPT_TEMPLATE = """You are extracting supply chain information from a company web page.

The page is from the domain: {domain}

Page text:
{text}

STEP 1 - RELEVANCE CHECK: First, determine whether the text above actually describes a real product for sale. If the text is a cookie-consent notice, navigation menu, legal disclaimer, login/account page, or otherwise NOT genuinely about a specific product, respond with "unknown" for ALL fields below.

STEP 2 - EXTRACTION: If the text does describe a real product, extract the following, based ONLY on what is explicitly stated or clearly implied in the text. Do not guess or use outside knowledge.

1. company: the name of the company that owns/operates this website
2. product: the name of the main product described on this page
3. manufactures: does the text indicate this company MANUFACTURES the product itself? Answer exactly "yes", "no", or "unknown".
4. distributed_by: if the product is made by a DIFFERENT company/brand than the one operating this site, name that other company. Otherwise "unknown".
5. made_in: the country or location where the product was made, if stated. Otherwise "unknown".
6. is_component_of: ONLY if this product is explicitly described as a component/part of a larger product, name that larger product using a short product name only (not a description). Otherwise "unknown".

For any field where the information is not present in the text, respond exactly with "unknown" - do not guess.

Respond in this exact JSON format:
{{"company": "...", "product": "...", "manufactures": "...", "distributed_by": "...", "made_in": "...", "is_component_of": "..."}}
"""

write_lock = threading.Lock()
backoff_lock = threading.Lock()
state = {"consecutive_errors": 0, "current_backoff": BACKOFF_START_SECONDS, "backoff_until": 0}


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def wait_if_backing_off():
    while True:
        with backoff_lock:
            remaining = state["backoff_until"] - time.time()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 5))


def record_error():
    with backoff_lock:
        state["consecutive_errors"] += 1
        if state["consecutive_errors"] >= CONSECUTIVE_ERROR_THRESHOLD:
            backoff = state["current_backoff"]
            print(f"[{now()}] {state['consecutive_errors']} consecutive errors - "
                  f"likely server instability. Backing off {backoff/60:.1f} min...")
            state["backoff_until"] = time.time() + backoff
            state["current_backoff"] = min(backoff * BACKOFF_MULTIPLIER, BACKOFF_MAX_SECONDS)


def record_success():
    with backoff_lock:
        if state["consecutive_errors"] > 0:
            print(f"[{now()}] Recovered after {state['consecutive_errors']} consecutive errors.")
        state["consecutive_errors"] = 0
        state["current_backoff"] = BACKOFF_START_SECONDS


def parse_json_response(raw):
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    return json.loads(cleaned)


def call_model(model, domain, text):
    text = text[:3000]
    prompt = PROMPT_TEMPLATE.format(domain=domain, text=text)
    for attempt in range(MAX_RETRIES):
        wait_if_backing_off()
        try:
            resp = requests.post(
                ENDPOINT,
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0},
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]
            parsed = parse_json_response(raw)
            record_success()
            return parsed, None
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 * (attempt + 1))
                continue
            record_error()
            return None, str(e)
    record_error()
    return None, "max retries exceeded"


def process_row(model, row):
    parsed, error = call_model(model, row["domain"], row["text"])
    if error:
        return {"domain": row["domain"], "url": row["url"], "status": "error", "error": error}
    return {"domain": row["domain"], "url": row["url"], "status": "extracted", **parsed}


def main(model, input_jsonl, output_jsonl, skiplog_path):
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
    print(f"[{now()}] Model: {model}")
    print(f"[{now()}] Already have {len(already_done):,} domains in output - will skip those.")

    rows_to_process = []
    with open(input_jsonl, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["domain"] not in already_done:
                rows_to_process.append(r)

    total = len(rows_to_process)
    print(f"[{now()}] Processing {total:,} remaining domains with {MAX_WORKERS} workers...")

    extracted = 0
    errors = 0
    completed = 0

    out_mode = "a" if output_path.exists() else "w"
    log_mode = "a" if skiplog_path.exists() else "w"

    with open(output_path, out_mode, encoding="utf-8") as outfile, \
         open(skiplog_path, log_mode, encoding="utf-8") as skiplog, \
         ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        futures = {executor.submit(process_row, model, row): row for row in rows_to_process}

        for future in as_completed(futures):
            result = future.result()
            completed += 1

            with write_lock:
                if result["status"] == "extracted":
                    outfile.write(json.dumps(result) + "\n")
                    outfile.flush()
                    extracted += 1
                else:
                    skiplog.write(f"ERROR\tdomain={result['domain']}\turl={result['url']}\treason={result.get('error')}\n")
                    errors += 1

                if completed % PROGRESS_INTERVAL == 0:
                    print(f"[{now()}] Processed {completed:,}/{total:,} | extracted={extracted:,} errors={errors:,}")

    print(f"\n[{now()}] --- Done ---")
    print(f"Extracted: {extracted:,}")
    print(f"Errors:    {errors:,}")


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "mistral-medium-3.5-128b"
    input_jsonl = sys.argv[2] if len(sys.argv) > 2 else "data/interim/fetched_pages.jsonl"
    output_jsonl = sys.argv[3] if len(sys.argv) > 3 else f"data/interim/extraction_results_{model}.jsonl"
    skiplog_path = sys.argv[4] if len(sys.argv) > 4 else f"data/interim/extraction_skipped_{model}.log"
    main(model, input_jsonl, output_jsonl, skiplog_path)
