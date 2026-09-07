"""
Build the remaining-domains list for the single-worker Stage B resume:
- Exclude domains already successfully fetched (found=true) across the 4 shards
- Exclude domains confirmed genuinely NOT_FOUND in the index (trustworthy - not
  caused by the block, since NOT_FOUND means the CDX server responded normally
  and simply had no record)
- Keep everything else (including CDX_ERROR entries) for retry, since those
  failures were caused by the IP block, not because the URL doesn't exist

Also seeds the final fetched_pages.jsonl with the already-successful records.
"""
import json
from pathlib import Path

already_found_domains = set()
not_found_domains = set()

# Collect successes from all 4 shards, and seed the final output file
with open("data/interim/fetched_pages.jsonl", "w", encoding="utf-8") as out:
    for i in range(4):
        shard_file = Path(f"data/interim/fetched_pages_shard{i}.jsonl")
        if not shard_file.exists():
            continue
        with open(shard_file, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                already_found_domains.add(record["domain"])
                out.write(line if line.endswith("\n") else line + "\n")

# Collect genuinely not-found domains from all 4 shard logs
for i in range(4):
    log_file = Path(f"data/interim/stage_b_skipped_shard{i}.log")
    if not log_file.exists():
        continue
    with open(log_file, encoding="utf-8") as f:
        for line in f:
            if line.startswith("NOT_FOUND"):
                # format: NOT_FOUND\tdomain=X\turl=Y
                domain = line.split("\t")[1].replace("domain=", "").strip()
                not_found_domains.add(domain)

skip_set = already_found_domains | not_found_domains

# Build the remaining list
total = 0
remaining = 0
with open("data/interim/domain_urls.jsonl", encoding="utf-8") as f, \
     open("data/interim/domain_urls_remaining.jsonl", "w", encoding="utf-8") as out:
    for line in f:
        total += 1
        record = json.loads(line)
        if record["domain"] not in skip_set:
            out.write(line if line.endswith("\n") else line + "\n")
            remaining += 1

print(f"Already found (seeded into fetched_pages.jsonl): {len(already_found_domains):,}")
print(f"Genuinely not-found (excluded from retry):        {len(not_found_domains):,}")
print(f"Total domains in corpus:                           {total:,}")
print(f"Remaining to process:                               {remaining:,}")
