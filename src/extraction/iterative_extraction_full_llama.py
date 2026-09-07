"""
Production, full-corpus version of the iterative (per-relation) extraction
pipeline, adapted from Carta et al. (2023).

Matches the robustness features of the original single-pass pipeline
(run_extraction.py): concurrent workers, domain-skip resume, incremental
checkpointed writes, and consecutive-error backoff.

Usage:
    EXTRACTION_ENDPOINT=http://localhost:8003/v1/chat/completions \
    python3 iterative_extraction_full.py \
        data/interim/fetched_pages.jsonl \
        data/interim/extraction_results_iterative_full.jsonl \
        data/interim/extraction_skipped_iterative_full.log
"""

import json
import os
import re
import sys
import threading
import time
from datetime import datetime

import requests

ENDPOINT = os.environ.get("EXTRACTION_ENDPOINT", "http://localhost:8003/v1/chat/completions")
MODEL_NAME = "lambdalabs/Llama-3.3-70B-Instruct-AWQ-4bit"
TEXT_TRUNCATE = 3000
MAX_WORKERS = 20
BACKOFF_START_SECONDS = 30
BACKOFF_MULTIPLIER = 2
BACKOFF_MAX_SECONDS = 600
CONSECUTIVE_ERROR_THRESHOLD = 5

ENTITY_PROMPT_TEMPLATE = """You are extracting information from a company web page.
The page is from the domain: {domain}
Page text:
{text}

STEP 1 - RELEVANCE CHECK: First, determine whether the text above actually describes a real product for sale. If the text is a cookie-consent notice, navigation menu, legal disclaimer, login/account page, or otherwise NOT genuinely about a specific product, respond with "unknown" for both fields below.

STEP 2 - EXTRACTION: If the text does describe a real product, extract:
1. company: the name of the company that owns/operates this website
2. product: the name of the main product described on this page

Respond in this exact JSON format:
{{"company": "...", "product": "..."}}
"""

RELATION_PROMPTS = {
    "manufactures": """You are extracting supply chain information from a company web page.
Company: {company}
Product: {product}
Page text:
{text}

Based ONLY on what is explicitly stated in the text, does the text indicate that {company} MANUFACTURES {product} itself?
Answer with exactly one word: "yes", "no", or "unknown". Do not guess -- if the text does not explicitly address this, answer "unknown".
""",
    "distributed_by": """You are extracting supply chain information from a company web page.
Company: {company}
Product: {product}
Page text:
{text}

Based ONLY on what is explicitly stated in the text, if {product} is made by a DIFFERENT company or brand than {company} (the company operating this website), name that other company/brand.
If no different company/brand is explicitly named, or if {company} makes the product itself, respond with exactly: unknown
Respond with ONLY the company/brand name, or "unknown". Do not guess.
""",
    "made_in": """You are extracting supply chain information from a company web page.
Company: {company}
Product: {product}
Page text:
{text}

Based ONLY on what is explicitly stated in the text, what country or location is stated as where {product} was MANUFACTURED?
Do NOT infer this from: a company's business/mailing address, the origin of a raw material or ingredient, or shipping/fulfillment origin. Only extract this if the text explicitly describes where the physical product itself was produced, crafted, assembled, or manufactured.
IMPORTANT: Always respond in ENGLISH, even if the source text is in another language (e.g. translate "Spanien" or "Espagne" to "Spain").
Respond with ONLY the country/location, or exactly "unknown" if not explicitly stated. Do not guess.
""",
    "is_component_of": """You are extracting supply chain information from a company web page.
Company: {company}
Product: {product}
Page text:
{text}

Based ONLY on what is explicitly stated in the text, is {product} explicitly described as a replacement part, spare part, or accessory FOR one specific, clearly named parent product (e.g. "Replacement Blade for XYZ Chainsaw Model 100" -> XYZ Chainsaw Model 100)?
Do NOT extract this if the compatible product is only a generic category (e.g. "fits most rotary tables" or "for Toyota vehicles") rather than one specific named model.
IMPORTANT: Respond with ONLY the short name of the parent product (a few words), never a full sentence or explanation. If not applicable, respond with exactly: unknown
""",
}

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
            print(f"[{now()}] {state['consecutive_errors']} consecutive errors - backing off {backoff/60:.1f} min...")
            state["backoff_until"] = time.time() + backoff
            state["current_backoff"] = min(backoff * BACKOFF_MULTIPLIER, BACKOFF_MAX_SECONDS)


def record_success():
    with backoff_lock:
        if state["consecutive_errors"] > 0:
            print(f"[{now()}] Recovered after {state['consecutive_errors']} consecutive errors.")
        state["consecutive_errors"] = 0
        state["current_backoff"] = BACKOFF_START_SECONDS


def call_llm(prompt, max_tokens=200):
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    resp = requests.post(ENDPOINT, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def parse_json_response(raw):
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    return json.loads(cleaned)


def clean_relation_response(raw):
    cleaned = raw.strip().strip('"').strip("'").strip(".")
    return cleaned if cleaned else "unknown"


def process_page(domain, text):
    wait_if_backing_off()
    text = text[:TEXT_TRUNCATE]

    try:
        entity_prompt = ENTITY_PROMPT_TEMPLATE.format(domain=domain, text=text)
        raw = call_llm(entity_prompt, max_tokens=150)
        entities = parse_json_response(raw)
        company = entities.get("company", "unknown")
        product = entities.get("product", "unknown")
    except Exception as e:
        record_error()
        return {"domain": domain, "status": "error", "error": f"entity_extraction: {e}"}

    if not company or not product or str(company).strip().lower() == "unknown" or str(product).strip().lower() == "unknown":
        record_success()
        return {
            "domain": domain, "status": "extracted", "company": company, "product": product,
            "manufactures": "unknown", "distributed_by": "unknown",
            "made_in": "unknown", "is_component_of": "unknown",
        }

    result = {"domain": domain, "status": "extracted", "company": company, "product": product}
    had_error = False

    for field, template in RELATION_PROMPTS.items():
        prompt = template.format(company=company, product=product, text=text)
        try:
            raw = call_llm(prompt, max_tokens=60)
            result[field] = clean_relation_response(raw)
        except Exception:
            result[field] = "unknown"
            had_error = True

    if had_error:
        record_error()
    else:
        record_success()

    return result


def load_done_domains(output_path):
    done = set()
    if os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("domain"):
                        done.add(rec["domain"])
                except json.JSONDecodeError:
                    continue
    return done


def main():
    if len(sys.argv) != 4:
        print("Usage: python3 iterative_extraction_full.py <input.jsonl> <output.jsonl> <skipped.log>")
        sys.exit(1)

    input_path, output_path, skipped_log_path = sys.argv[1], sys.argv[2], sys.argv[3]

    print(f"[{now()}] Model: {MODEL_NAME} (iterative, 5 calls/page)")

    done_domains = load_done_domains(output_path)
    print(f"[{now()}] Already have {len(done_domains)} domains in output - will skip those.")

    pages = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            domain = rec.get("domain", "")
            text = rec.get("text") or rec.get("archived_page_text", "")
            if domain and text and domain not in done_domains:
                pages.append((domain, text))

    print(f"[{now()}] Processing {len(pages):,} remaining domains with {MAX_WORKERS} workers...")

    out_f = open(output_path, "a", encoding="utf-8")
    skip_f = open(skipped_log_path, "a", encoding="utf-8")

    n_extracted = 0
    n_errors = 0
    n_processed = 0
    t0 = time.time()

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_page, domain, text): domain for domain, text in pages}
        for future in as_completed(futures):
            domain = futures[future]
            try:
                result = future.result()
            except Exception as e:
                result = {"domain": domain, "status": "error", "error": str(e)}

            with write_lock:
                if result.get("status") == "extracted":
                    n_extracted += 1
                    out_f.write(json.dumps(result) + "\n")
                    out_f.flush()
                else:
                    n_errors += 1
                    skip_f.write(json.dumps(result) + "\n")
                    skip_f.flush()
                n_processed += 1

                if n_processed % 100 == 0:
                    print(f"[{now()}] Processed {n_processed:,}/{len(pages):,} | extracted={n_extracted:,} errors={n_errors:,}")

    out_f.close()
    skip_f.close()

    elapsed = time.time() - t0
    print(f"\n[{now()}] --- Done ---")
    print(f"Extracted: {n_extracted:,}")
    print(f"Errors:    {n_errors:,}")
    print(f"Elapsed:   {elapsed/3600:.2f} hours")


if __name__ == "__main__":
    main()
