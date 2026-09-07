"""
Diagnose why Athena's match rate is lower than the CDX API's for the same
domains - likely a URL string format mismatch, not genuinely missing data.
"""
import json
import csv

# Load what we requested
requested = {}
with open("data/interim/domain_urls_scattered1k.jsonl") as f:
    for line in f:
        r = json.loads(line)
        requested[r["url"]] = r["domain"]

# Load what Athena matched
matched_urls = set()
with open("data/interim/athena_lookup_test.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        matched_urls.add(row["url"])

missed = [u for u in requested if u not in matched_urls]
print(f"Total requested: {len(requested)}")
print(f"Matched: {len(matched_urls)}")
print(f"Missed: {len(missed)}")
print(f"\nFirst 5 missed URLs (exact strings, for comparison):")
for u in missed[:5]:
    print(f"  {u!r}")
