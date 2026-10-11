import csv
import json
from pathlib import Path

evidence_file = Path(
    "data/interim/eval/figures/fig_kg_example_evidence.csv"
)
pages_file = Path("data/interim/fetched_pages.jsonl")
output = Path("data/interim/eval/figures/fig_kg_example_review.csv")

with evidence_file.open(encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    original_columns = reader.fieldnames
    rows = list(reader)

wanted = {r["source_url"] for r in rows if r["source_url"]}
matches = {}
bad_lines = 0
duplicate_matches = 0

# Exact URL matching; no guessed domain-level substitutions.
with pages_file.open(encoding="utf-8") as f:
    for line in f:
        try:
            page = json.loads(line)
        except json.JSONDecodeError:
            bad_lines += 1
            continue

        url = page.get("url")
        if url not in wanted:
            continue

        text = page.get("text")
        if not isinstance(text, str) or not text.strip():
            continue

        if url in matches:
            duplicate_matches += 1
            continue

        matches[url] = page

extra_columns = [
    "text_match_status",
    "domain",
    "page_status",
    "input_text_first_3000",
    "full_archived_text",
    "supporting_excerpt",
    "verdict",
    "checked_by",
    "review_note",
]

with output.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=original_columns + extra_columns
    )
    writer.writeheader()

    for row in rows:
        page = matches.get(row["source_url"])
        text = page.get("text", "") if page else ""

        row.update({
            "text_match_status": (
                "exact_url_match" if page else "missing_text"
            ),
            "domain": page.get("domain", "") if page else "",
            "page_status": page.get("status", "") if page else "",
            "input_text_first_3000": text[:3000],
            "full_archived_text": text,
            "supporting_excerpt": "",
            "verdict": "",
            "checked_by": "",
            "review_note": (
                "Graph-derived seller–product link; verify listing "
                "and operator identity."
                if row["relation"] == "offers"
                else ""
            ),
        })
        writer.writerow(row)

missing = sorted(wanted - matches.keys())

print(f"Saved: {output}")
print(f"Evidence rows: {len(rows)}")
print(f"Unique source URLs matched: {len(matches)} / {len(wanted)}")
print(f"Unparseable page lines: {bad_lines}")
print(f"Additional matching records: {duplicate_matches}")
print("Verdicts are blank: this script collects evidence, not verification.")

for url in missing:
    print(f"MISSING TEXT: {url}")
