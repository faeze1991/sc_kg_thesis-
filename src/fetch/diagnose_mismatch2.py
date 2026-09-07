"""
Broader diagnostic: after the trailing-slash fix, sample MULTIPLE remaining
misses and categorize the mismatch type for each, rather than generalizing
from a single example.
"""
import json
import csv

requested = {}
with open("data/interim/domain_urls_scattered1k.jsonl") as f:
    for line in f:
        r = json.loads(line)
        requested[r["url"]] = r["domain"]

matched_urls = set()
with open("data/interim/athena_lookup_test2.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        matched_urls.add(row["url"])

missed = [u for u in requested if u not in matched_urls]
print(f"Total requested: {len(requested)}")
print(f"Matched (after trailing-slash fix): {len(matched_urls)}")
print(f"Still missed: {len(missed)}")
print(f"\n--- Sample of 10 still-missed URLs, with domain ---")
for u in missed[:10]:
    print(f"  domain={requested[u]}")
    print(f"  url={u!r}")
    print()
