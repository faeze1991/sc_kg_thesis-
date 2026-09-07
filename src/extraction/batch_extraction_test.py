"""
Larger-scale prompt test: 40 pages x 3 models, with proper JSON parsing
and summary statistics - signal rate, format consistency, model agreement.
"""
import json
import os
import re
import requests
import time

API_KEY = os.environ["GWDG_API_KEY"]
ENDPOINT = "https://chat-ai.academiccloud.de/v1/chat/completions"

MODELS = [
    "mistral-medium-3.5-128b",
    "qwen3-30b-a3b-instruct-2507",
    "openai-gpt-oss-120b",
]

SAMPLE_SIZE = 40

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


def call_model(model, domain, text):
    text = text[:3000]
    prompt = PROMPT_TEMPLATE.format(domain=domain, text=text)
    resp = requests.post(
        ENDPOINT,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0},
        timeout=90,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def parse_json_response(raw):
    """Strip markdown code fences if present, then parse JSON."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
    return json.loads(cleaned)


def main():
    samples = []
    with open("data/interim/fetched_pages.jsonl") as f:
        for i, line in enumerate(f):
            if i >= SAMPLE_SIZE:
                break
            samples.append(json.loads(line))

    print(f"Testing {len(samples)} pages x {len(MODELS)} models = {len(samples)*len(MODELS)} calls...")

    all_results = []
    stats = {m: {"parse_ok": 0, "parse_fail": 0, "any_signal": 0, "all_unknown": 0} for m in MODELS}

    for idx, s in enumerate(samples):
        row_result = {"domain": s["domain"], "url": s["url"], "models": {}}
        for model in MODELS:
            try:
                raw = call_model(model, s["domain"], s["text"])
                parsed = parse_json_response(raw)
                stats[model]["parse_ok"] += 1
                signal_fields = ["manufactures", "distributed_by", "made_in", "is_component_of"]
                has_signal = any(
                    str(parsed.get(f, "unknown")).lower() not in ("unknown", "no")
                    for f in signal_fields
                )
                if has_signal:
                    stats[model]["any_signal"] += 1
                else:
                    stats[model]["all_unknown"] += 1
                row_result["models"][model] = parsed
            except Exception as e:
                stats[model]["parse_fail"] += 1
                row_result["models"][model] = {"error": str(e)}
        all_results.append(row_result)
        print(f"  [{idx+1}/{len(samples)}] {s['domain']}")

    # Save raw results for later inspection
    with open("data/interim/batch_extraction_test_results.json", "w") as f:
     json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for model in MODELS:
        s = stats[model]
        total = s["parse_ok"] + s["parse_fail"]
        print(f"\n{model}:")
        print(f"  JSON parse success: {s['parse_ok']}/{total} ({100*s['parse_ok']/total:.1f}%)")
        if s["parse_ok"] > 0:
            print(f"  Pages with real signal: {s['any_signal']}/{s['parse_ok']} ({100*s['any_signal']/s['parse_ok']:.1f}%)")
            print(f"  Pages all-unknown: {s['all_unknown']}/{s['parse_ok']} ({100*s['all_unknown']/s['parse_ok']:.1f}%)")

    print(f"\nFull results saved to: data/interim/batch_extraction_test_results.json")


if __name__ == "__main__":
    main()
