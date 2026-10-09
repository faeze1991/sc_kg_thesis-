import csv, json, random, sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 183
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 42

PAGES = "data/interim/fetched_pages.jsonl"
USED = ["data/interim/annotation_sample_BLIND.csv", "data/interim/annotation_topup_BLIND.csv"]
OUT = "data/interim/annotation_new_BLIND.csv"

used = set()
for path in USED:
    with open(path, newline="", encoding="utf-8") as f:
        used.update(r["domain"].strip() for r in csv.DictReader(f))
print(f"[INFO] {len(used)} previously annotated domains excluded")

pool = []
with open(PAGES, encoding="utf-8") as f:
    for line in f:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("status") == "found" and r.get("text") and r["domain"] not in used:
            pool.append(r)
print(f"[INFO] {len(pool)} eligible pages")

random.seed(SEED)
sample = random.sample(pool, N)

cols = ["record_id", "domain", "url", "product", "company", "sampled_for_field",
        "gold_manufactures_yes_no", "gold_distributed_by", "gold_made_in",
        "gold_is_component_of", "annotator_notes", "archived_page_text"]
with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for i, r in enumerate(sample):
        w.writerow({"record_id": f"new_{i:04d}", "domain": r["domain"], "url": r["url"],
                    "product": "", "company": "", "sampled_for_field": "random",
                    "archived_page_text": r["text"]})
print(f"[DONE] wrote {len(sample)} rows to {OUT}")
