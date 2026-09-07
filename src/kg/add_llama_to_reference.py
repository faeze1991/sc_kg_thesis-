"""
add_llama_to_reference.py

Adds Llama 3.3 70B's predictions (from the 2,000-page benchmark run) into
data/interim/annotation_reference_HIDDEN.jsonl, matched by domain, so it
can be scored alongside Qwen/gpt-oss/mistral using the existing
score_against_claude_annotation.py script.

Usage:
    python3 src/kg/add_llama_to_reference.py
"""

import json

REFERENCE_PATH = "data/interim/annotation_reference_HIDDEN.jsonl"
LLAMA_RESULTS_PATH = "data/interim/extraction_results_llama70b_bench2k.jsonl"

RELATION_FIELDS = ["manufactures", "distributed_by", "made_in", "is_component_of"]


def main():
    print("[INFO] Loading Llama 3.3 70B extraction results...")
    llama_by_domain = {}
    with open(LLAMA_RESULTS_PATH) as f:
        for line in f:
            rec = json.loads(line)
            domain = rec.get("domain")
            if domain:
                llama_by_domain[domain] = {f: rec.get(f, "unknown") for f in RELATION_FIELDS}
    print(f"[INFO] Loaded {len(llama_by_domain)} Llama records.")

    print("[INFO] Loading existing reference file...")
    entries = []
    with open(REFERENCE_PATH) as f:
        for line in f:
            entries.append(json.loads(line))
    print(f"[INFO] Loaded {len(entries)} reference records.")

    n_matched = 0
    for entry in entries:
        domain = entry.get("domain")
        llama_pred = llama_by_domain.get(domain)
        entry["llama70b"] = llama_pred
        if llama_pred is not None:
            n_matched += 1

    print(f"[INFO] Matched Llama predictions for {n_matched}/{len(entries)} reference records.")

    with open(REFERENCE_PATH, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    print(f"[INFO] Updated {REFERENCE_PATH} with llama70b predictions.")


if __name__ == "__main__":
    main()
