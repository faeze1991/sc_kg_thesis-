"""
add_archived_text_to_annotation.py

Fixes a methodological issue: annotation should be done against the ARCHIVED
Common Crawl page text (what the model actually saw at extraction time), not
the live current website, which may have changed or gone dead since the
CC-MAIN-2020-40 crawl.

Reads the existing annotation_sample_BLIND.csv and adds a new column,
'archived_page_text', pulled from data/interim/fetched_pages_bench2k.jsonl
(and fetched_pages_mistral300.jsonl as a fallback, in case some domains are
only in that file). Overwrites annotation_sample_BLIND.csv with the new
column added -- your existing gold_* entries (if you've already started
annotating) are preserved untouched.

Usage:
    python3 src/kg/add_archived_text_to_annotation.py
"""

import json
import csv

BENCH2K_PATH = "data/interim/fetched_pages_bench2k.jsonl"
MISTRAL300_PATH = "data/interim/fetched_pages_mistral300.jsonl"
ANNOTATION_CSV = "data/interim/annotation_sample_BLIND.csv"

# Common field name candidates for the extracted text in your Stage B output --
# adjust if your fetched_pages files use a different key name.
TEXT_FIELD_CANDIDATES = ["text", "extracted_text", "content", "page_text", "body_text"]


def load_text_by_domain(path):
    texts = {}
    try:
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                domain = rec.get("domain")
                if not domain:
                    continue
                for field in TEXT_FIELD_CANDIDATES:
                    if field in rec and rec[field]:
                        texts[domain] = rec[field]
                        break
    except FileNotFoundError:
        print(f"[WARN] {path} not found -- skipping.")
    return texts


def main():
    print("[INFO] Loading archived page text...")
    texts = load_text_by_domain(BENCH2K_PATH)
    print(f"[INFO] Loaded {len(texts)} archived texts from {BENCH2K_PATH}")

    fallback_texts = load_text_by_domain(MISTRAL300_PATH)
    n_added_from_fallback = 0
    for domain, text in fallback_texts.items():
        if domain not in texts:
            texts[domain] = text
            n_added_from_fallback += 1
    print(f"[INFO] Added {n_added_from_fallback} more from {MISTRAL300_PATH}")

    if not texts:
        print("[ERROR] No archived text loaded. Check that TEXT_FIELD_CANDIDATES "
              "matches the actual field name in your fetched_pages files. "
              "Run: head -1 data/interim/fetched_pages_bench2k.jsonl to check field names.")
        return

    print(f"\n[INFO] Reading existing annotation CSV: {ANNOTATION_CSV}")
    rows = []
    with open(ANNOTATION_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    n_matched = 0
    n_missing = 0
    for row in rows:
        domain = row.get("domain", "")
        archived = texts.get(domain, "")
        if archived:
            n_matched += 1
        else:
            n_missing += 1
        row["archived_page_text"] = archived

    print(f"[INFO] Matched archived text for {n_matched}/{len(rows)} rows")
    if n_missing:
        print(f"[WARN] {n_missing} rows have NO archived text available -- "
              f"for these, you'll need to note in annotator_notes that no "
              f"archived text was found and annotate 'unknown' rather than "
              f"relying on the live website.")

    new_fieldnames = fieldnames + ["archived_page_text"] if "archived_page_text" not in fieldnames else fieldnames

    with open(ANNOTATION_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[INFO] Updated {ANNOTATION_CSV} with 'archived_page_text' column.")
    print("[IMPORTANT] Annotate using the 'archived_page_text' column content, "
          "NOT by visiting the live url -- the live site may have changed since "
          "the 2020 crawl. Only fall back to visiting the live url for extra "
          "context when archived_page_text is empty, and note this in "
          "annotator_notes when you do.")


if __name__ == "__main__":
    main()
