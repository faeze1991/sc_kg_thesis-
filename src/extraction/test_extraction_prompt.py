"""
Stage C prompt test: extract company/product/relation info from a handful
of already-fetched pages, using GWDG's API, comparing 3 candidate models
side by side on the exact same real pages.
"""
import json
import os
import requests

API_KEY = os.environ["GWDG_API_KEY"]
ENDPOINT = "https://chat-ai.academiccloud.de/v1/chat/completions"

MODELS = [
    "mistral-medium-3.5-128b",
    "qwen3-30b-a3b-instruct-2507",
    "openai-gpt-oss-120b",
]

PROMPT_TEMPLATE = """You are extracting supply chain information from a company web page.

The page is from the domain: {domain}

Page text:
{text}

Extract the following, based ONLY on what is explicitly stated or clearly implied in the text above. Do not guess or use outside knowledge.

1. company: the name of the company that owns/operates this website
2. product: the name of the main product described on this page
3. manufactures: does the text indicate this company MANUFACTURES the product itself? (not just sells it)
4. distributed_by: if the product is made by a DIFFERENT company/brand than the one operating this site, name that other company
5. made_in: the country or location where the product was made, if stated
6. is_component_of: if this product is described as a component/part of a larger product, name that larger product

For any field where the information is not present in the text, respond exactly with "unknown" - do not guess.

Respond in this exact JSON format:
{{"company": "...", "product": "...", "manufactures": "...", "distributed_by": "...", "made_in": "...", "is_component_of": "..."}}
"""


def call_model(model, domain, text):
    # Truncate to keep this first test's prompt size reasonable
    text = text[:3000]
    prompt = PROMPT_TEMPLATE.format(domain=domain, text=text)
    resp = requests.post(
        ENDPOINT,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def main():
    samples = []
    with open("data/interim/fetched_pages.jsonl") as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            samples.append(json.loads(line))

    for s in samples:
        print(f"\n{'='*70}")
        print(f"DOMAIN: {s['domain']}")
        print(f"URL: {s['url']}")
        print(f"{'='*70}")
        for model in MODELS:
            print(f"\n--- {model} ---")
            try:
                result = call_model(model, s["domain"], s["text"])
                print(result)
            except Exception as e:
                print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
