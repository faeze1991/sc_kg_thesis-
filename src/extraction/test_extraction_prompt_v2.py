"""
Stage C prompt test v2: adds a relevance-check step (reject cookie banners,
nav menus, etc.) and tightens output format (manufactures as explicit
yes/no/unknown; is_component_of restricted to a short product name).
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
    target_domains = {"hobbyboden.dk", "feng-shui-architects.com", "titaniumdynamics.com",
                       "deniswhitehomecinema.com.au", "ifairings.com"}
    samples = []
    with open("data/interim/fetched_pages.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["domain"] in target_domains:
                samples.append(r)
            if len(samples) == len(target_domains):
                break

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
