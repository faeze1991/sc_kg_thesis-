"""
Deduplicate Athena lookup results: some URLs matched multiple index entries
(e.g. both '/page' and '/page/' existing as separate crawled captures, both
matching after trailing-slash normalization). Keep exactly one row per
unique input URL.

Usage (from the project root):
    python -m src.fetch.dedupe_athena_results \
        data/interim/athena_lookup_results_all.csv data/interim/athena_lookup_results_all_dedup.csv
"""
import csv
import sys

input_csv = sys.argv[1]
output_csv = sys.argv[2]

seen = set()
total_rows = 0
written = 0
duplicates_dropped = 0

with open(input_csv, encoding="utf-8") as fin, \
     open(output_csv, "w", encoding="utf-8", newline="") as fout:
    reader = csv.DictReader(fin)
    writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
    writer.writeheader()
    for row in reader:
        total_rows += 1
        url = row["url"]
        if url in seen:
            duplicates_dropped += 1
            continue
        seen.add(url)
        writer.writerow(row)
        written += 1

print(f"Total rows read:       {total_rows:,}")
print(f"Duplicate rows dropped: {duplicates_dropped:,}")
print(f"Unique rows written:   {written:,}")
