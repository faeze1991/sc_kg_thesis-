"""
Export documented edge cases (very short / very long extracted text) to a
separate file for thesis documentation and appendix reference. These
records REMAIN in the main fetched_pages.jsonl untouched - this is purely
an additional reference export, not a removal.
"""
import json
from pathlib import Path

SMALL_THRESHOLD = 200
LARGE_THRESHOLD = 100000

Path("data/interim/edge_cases").mkdir(parents=True, exist_ok=True)

small_cases = []
large_cases = []

with open("data/interim/fetched_pages.jsonl") as f:
    for line in f:
        r = json.loads(line)
        length = len(r["text"])
        if length < SMALL_THRESHOLD:
            small_cases.append(r)
        elif length > LARGE_THRESHOLD:
            large_cases.append(r)

with open("data/interim/edge_cases/small_text_pages.jsonl", "w") as f:
    for r in small_cases:
        f.write(json.dumps(r) + "\n")

with open("data/interim/edge_cases/large_text_pages.jsonl", "w") as f:
    for r in large_cases:
        # For large pages, save only a preview + metadata, not the full 900k+ chars,
        # to keep this reference file manageable - full text still safely in main corpus
        preview_record = {**r, "text": r["text"][:2000], "full_text_length": len(r["text"]),
                           "note": "text truncated to 2000 chars for this reference file; full text remains in fetched_pages.jsonl"}
        f.write(json.dumps(preview_record) + "\n")

print(f"Small-text edge cases (<{SMALL_THRESHOLD} chars): {len(small_cases)} -> data/interim/edge_cases/small_text_pages.jsonl")
print(f"Large-text edge cases (>{LARGE_THRESHOLD} chars): {len(large_cases)} -> data/interim/edge_cases/large_text_pages.jsonl")
print(f"\nBoth categories REMAIN in the main fetched_pages.jsonl unchanged - these are reference exports only.")
