"""
iterative_extraction.py

Iterative extraction pipeline adapted from Carta et al. (2023), "Iterative
Zero-Shot LLM Prompting for Knowledge Graph Construction" (arXiv:2307.01128).

ADAPTATION RATIONALE (documented for thesis methodology section):
Carta et al.'s core mechanism decomposes complex single-pass extraction into
focused iterative steps, looping ONE ENTITY AT A TIME over an open,
large entity set (their tourism-webpage dataset had 27-62 entities per
document) to keep each individual LLM call simple and reliable -- their
paper explicitly documents that GPT-3.5 "fails at sticking to the complete
set" when asked to extract relations for many entities simultaneously.

This thesis's schema is structurally different: a FIXED, SMALL set of 4
relation types (manufactures, distributed_by, made_in, is_component_of),
with typically only 1-2 entities (company, product) per page -- there is no
large open entity set to iterate over the way Carta et al.'s method requires.

The adaptation applied here preserves the core decomposition PRINCIPLE while
substituting the correct axis of complexity for this task: instead of
iterating per-entity, this pipeline iterates PER RELATION TYPE, since the
relation schema (not the entity set) is the dimension of complexity in this
task. This mirrors Carta et al.'s "Iterative Triplet Extraction" step
(Section 4.3.1) exactly, with relations substituted for entities as the
loop variable.

Pipeline (5 calls per page):
    Call 1 (Entity Extraction):    extract company + product only
    Call 2 (Relation, focused):    manufactures only
    Call 3 (Relation, focused):    distributed_by only
    Call 4 (Relation, focused):    made_in only
    Call 5 (Relation, focused):    is_component_of only

Usage:
    EXTRACTION_ENDPOINT=http://localhost:8000/v1/chat/completions \
    python3 iterative_extraction.py \
        <input_pages.jsonl> <output.jsonl>

Input format: JSONL with at least {"domain": ..., "text": ...} per line
(or reuse the same schema as fetched_pages.jsonl / archived_page_text field).
"""

import json
import os
import re
import sys
import time
import requests

ENDPOINT = os.environ.get("EXTRACTION_ENDPOINT", "http://localhost:8000/v1/chat/completions")
MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
TEXT_TRUNCATE = 3000  # same truncation as the original single-pass prompt, for fair comparison

# --- Call 1: Entity Extraction (mirrors Carta et al. Section 4.3.1 "Entity Extraction") ---
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

# --- Calls 2-5: Iterative Relation Extraction, one focused call per relation type ---
# (mirrors Carta et al. Section 4.3.1 "Iterative Triplet Extraction" -- looping
#  over the relation schema instead of an open entity set)

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
Respond with ONLY the country/location, or exactly "unknown" if not explicitly stated. Do not guess.
""",

    "is_component_of": """You are extracting supply chain information from a company web page.
Company: {company}
Product: {product}
Page text:
{text}

Based ONLY on what is explicitly stated in the text, is {product} explicitly described as a replacement part, spare part, or accessory FOR one specific, clearly named parent product (e.g. "Replacement Blade for XYZ Chainsaw Model 100" -> XYZ Chainsaw Model 100)?
Do NOT extract this if the compatible product is only a generic category (e.g. "fits most rotary tables" or "for Toyota vehicles") rather than one specific named model.
Respond with ONLY the short name of the parent product, or exactly "unknown" if not applicable. Do not guess.
""",
}


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
    """Strip quotes/punctuation noise from a short relation-field response."""
    cleaned = raw.strip().strip('"').strip("'").strip(".")
    return cleaned if cleaned else "unknown"


def process_page(domain, text):
    text = text[:TEXT_TRUNCATE]

    # Call 1: Entity Extraction
    entity_prompt = ENTITY_PROMPT_TEMPLATE.format(domain=domain, text=text)
    try:
        raw = call_llm(entity_prompt, max_tokens=150)
        entities = parse_json_response(raw)
        company = entities.get("company", "unknown")
        product = entities.get("product", "unknown")
    except Exception as e:
        return {"domain": domain, "status": "error", "error": f"entity_extraction: {e}"}

    if not company or not product or str(company).strip().lower() == "unknown" or str(product).strip().lower() == "unknown":
        # relevance-check equivalent: nothing to extract relations about
        return {
            "domain": domain, "status": "extracted", "company": company, "product": product,
            "manufactures": "unknown", "distributed_by": "unknown",
            "made_in": "unknown", "is_component_of": "unknown",
        }

    result = {"domain": domain, "status": "extracted", "company": company, "product": product}

    # Calls 2-5: one focused call per relation type
    for field, template in RELATION_PROMPTS.items():
        prompt = template.format(company=company, product=product, text=text)
        try:
            raw = call_llm(prompt, max_tokens=60)
            result[field] = clean_relation_response(raw)
        except Exception as e:
            result[field] = f"ERROR:{e}"

    return result


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 iterative_extraction.py <input.jsonl> <output.jsonl>")
        print("Input JSONL must have 'domain' and 'text' (or 'archived_page_text') fields.")
        sys.exit(1)

    input_path, output_path = sys.argv[1], sys.argv[2]

    pages = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            domain = rec.get("domain", "")
            text = rec.get("text") or rec.get("archived_page_text", "")
            if domain and text:
                pages.append((domain, text))

    print(f"[INFO] Loaded {len(pages)} pages to process")
    print(f"[INFO] 5 calls per page = {len(pages) * 5} total LLM calls expected")

    results = []
    t0 = time.time()
    for i, (domain, text) in enumerate(pages):
        result = process_page(domain, text)
        results.append(result)
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"[{time.strftime('%H:%M:%S')}] Processed {i+1}/{len(pages)} | {elapsed:.0f}s elapsed")

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    n_ok = sum(1 for r in results if r.get("status") == "extracted")
    print(f"\n[DONE] {n_ok}/{len(results)} pages successfully extracted")
    print(f"[INFO] Saved to {output_path}")


if __name__ == "__main__":
    main()
