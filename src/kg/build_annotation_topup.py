"""
build_annotation_topup.py

HISTORICAL (earlier annotation round): this script drew the pages of the first
annotation round. Those pages were later labelled with LLM assistance and are
used only as "silver" training data for the classical baseline
(data/annotation/silver/). They are NOT part of the gold-standard evaluation,
which uses the 320 stratified pages (make_stratified_annotation_sheet.py).

Builds a TARGETED top-up annotation sample specifically for made_in and
is_component_of -- the two fields that were too sparse in the original
366-row sample (~11 and ~2 true positives respectively) to compute
meaningful precision/recall/F1.

Pulls from the FULL 280K corpus (not just the 2,000-page benchmark) since
these fields are rare (10.76% and 3.22% full-corpus rates for Qwen) --
there aren't enough positive cases in the smaller benchmark subset alone.

IMPORTANT LIMITATION (documented here and in output): only Qwen ran
extraction on the full 280K corpus. gpt-oss-20b and mistral only covered
the first ~2,000 / ~300 pages. So most top-up records will only have a
Qwen prediction to compare against -- gpt-oss/mistral coverage will be
partial (only for records that happen to also fall in their smaller
coverage). This means the original 366-row sample remains your primary
source for 3-way model comparison; this top-up is specifically for
getting statistically meaningful made_in/is_component_of quality numbers,
primarily against Qwen.

Excludes any domain already in the original 366-row sample, to avoid
duplicate annotation work.

Usage:
    python -m src.kg.build_annotation_topup
"""

import json
import random
import csv

random.seed(43)  # different seed from original sample (42) -- document in methodology

QWEN_NORMALIZED_PATH = "data/interim/extraction_results_normalized_FINAL.jsonl"
FETCHED_PAGES_FULL_PATH = "data/interim/fetched_pages.jsonl"
GPTOSS_PATH = "data/interim/extraction_results_gptoss20b_bench2k.jsonl"
MISTRAL_PATH = "data/interim/extraction_results_mistral_300.jsonl"
EXISTING_SAMPLE_PATH = "data/interim/annotation_sample_BLIND.csv"

OUT_BLIND_CSV = "data/interim/annotation_topup_BLIND.csv"
OUT_REFERENCE_JSONL = "data/interim/annotation_topup_reference_HIDDEN.jsonl"

N_PER_FIELD = 400
RELATION_FIELDS = ["manufactures", "distributed_by", "made_in", "is_component_of"]

# Text field candidates for the full fetched_pages.jsonl -- adjust if needed
TEXT_FIELD_CANDIDATES = ["text", "extracted_text", "content", "page_text", "body_text"]


def load_jsonl_by_domain(path, fields=None):
    records = {}
    try:
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                domain = rec.get("domain")
                if domain:
                    records[domain] = rec
    except FileNotFoundError:
        print(f"[WARN] {path} not found -- treating as empty.")
    return records


def load_existing_sample_domains(path):
    domains = set()
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=",")
            for row in reader:
                d = row.get("domain", "").strip()
                if d:
                    domains.add(d)
    except FileNotFoundError:
        print(f"[WARN] {path} not found -- proceeding without exclusion set.")
    return domains


def extract_predictions(rec):
    if rec is None:
        return None
    return {f: rec.get(f, "unknown") for f in RELATION_FIELDS}


def get_text_for_domain(domain, text_index):
    return text_index.get(domain, "")


def main():
    print("[INFO] Loading existing 366-row sample domains (to exclude duplicates)...")
    existing_domains = load_existing_sample_domains(EXISTING_SAMPLE_PATH)
    print(f"[INFO] {len(existing_domains)} domains already sampled, will exclude.")

    print("[INFO] Loading Qwen full corpus (this may take a moment)...")
    qwen_all = load_jsonl_by_domain(QWEN_NORMALIZED_PATH)
    print(f"[INFO] Loaded {len(qwen_all)} Qwen records.")

    gptoss_by_domain = load_jsonl_by_domain(GPTOSS_PATH)
    mistral_by_domain = load_jsonl_by_domain(MISTRAL_PATH)
    print(f"[INFO] gpt-oss coverage available for {len(gptoss_by_domain)} domains "
          f"(only useful for top-up records that overlap this set).")
    print(f"[INFO] mistral coverage available for {len(mistral_by_domain)} domains.")

    # Candidate pools: Qwen positive for made_in / is_component_of, excluding already-sampled
    made_in_pool = [
        d for d, r in qwen_all.items()
        if d not in existing_domains and str(r.get("made_in", "unknown")).strip().lower() != "unknown"
    ]
    component_pool = [
        d for d, r in qwen_all.items()
        if d not in existing_domains and str(r.get("is_component_of", "unknown")).strip().lower() != "unknown"
    ]
    print(f"[INFO] made_in positive pool (excluding existing sample): {len(made_in_pool)}")
    print(f"[INFO] is_component_of positive pool (excluding existing sample): {len(component_pool)}")

    n_made_in = min(N_PER_FIELD, len(made_in_pool))
    n_component = min(N_PER_FIELD, len(component_pool))
    if n_made_in < N_PER_FIELD:
        print(f"[WARN] Only {n_made_in} made_in-positive records available -- using all of them.")
    if n_component < N_PER_FIELD:
        print(f"[WARN] Only {n_component} is_component_of-positive records available -- using all of them.")

    made_in_sample = set(random.sample(made_in_pool, n_made_in))
    component_sample = set(random.sample(component_pool, n_component))

    # Union (a record can be positive for both fields -- keep it once, tag which field(s) it's for)
    all_selected = made_in_sample | component_sample
    all_selected = list(all_selected)
    random.shuffle(all_selected)
    print(f"[INFO] Total unique top-up records (union, some overlap expected): {len(all_selected)}")

    # Load full-corpus fetched page text
    print("[INFO] Loading full-corpus page text for archived_page_text column "
          "(this file is large, may take a moment)...")
    text_index = {}
    try:
        with open(FETCHED_PAGES_FULL_PATH) as f:
            for line in f:
                rec = json.loads(line)
                domain = rec.get("domain")
                if not domain or domain not in all_selected:
                    continue
                for field in TEXT_FIELD_CANDIDATES:
                    if field in rec and rec[field]:
                        text_index[domain] = rec[field]
                        break
    except FileNotFoundError:
        print(f"[WARN] {FETCHED_PAGES_FULL_PATH} not found -- archived_page_text will be empty.")

    print(f"[INFO] Matched archived text for {len(text_index)}/{len(all_selected)} selected domains.")

    # --- Write blind annotation CSV ---
    with open(OUT_BLIND_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow([
            "record_id", "domain", "url", "product", "company",
            "sampled_for_field",
            "gold_manufactures_yes_no", "gold_distributed_by",
            "gold_made_in", "gold_is_component_of", "annotator_notes",
            "archived_page_text"
        ])
        for i, domain in enumerate(all_selected):
            rec = qwen_all[domain]
            record_id = f"topup_{i:04d}"
            tags = []
            if domain in made_in_sample:
                tags.append("made_in")
            if domain in component_sample:
                tags.append("is_component_of")
            writer.writerow([
                record_id, domain, rec.get("url", ""),
                rec.get("product", ""), rec.get("company", ""),
                "+".join(tags),
                "", "", "", "", "",
                get_text_for_domain(domain, text_index),
            ])
    print(f"[INFO] Wrote blind top-up annotation CSV: {OUT_BLIND_CSV}")

    # --- Write hidden reference JSONL ---
    with open(OUT_REFERENCE_JSONL, "w", encoding="utf-8") as f:
        for i, domain in enumerate(all_selected):
            record_id = f"topup_{i:04d}"
            entry = {
                "record_id": record_id,
                "domain": domain,
                "qwen": extract_predictions(qwen_all.get(domain)),
                "gptoss20b": extract_predictions(gptoss_by_domain.get(domain)),  # likely None for most
                "mistral": extract_predictions(mistral_by_domain.get(domain)),  # likely None for most
            }
            f.write(json.dumps(entry) + "\n")
    print(f"[INFO] Wrote hidden reference file: {OUT_REFERENCE_JSONL}")

    n_gptoss_covered = sum(1 for d in all_selected if d in gptoss_by_domain)
    n_mistral_covered = sum(1 for d in all_selected if d in mistral_by_domain)
    print(f"\n[SUMMARY] Of {len(all_selected)} top-up records:")
    print(f"  - {n_made_in} tagged for made_in evaluation")
    print(f"  - {n_component} tagged for is_component_of evaluation")
    print(f"  - {n_gptoss_covered} also have a gpt-oss-20b prediction available")
    print(f"  - {n_mistral_covered} also have a mistral prediction available")
    print("\n[REMINDER] Do NOT open annotation_topup_reference_HIDDEN.jsonl while annotating.")


if __name__ == "__main__":
    main()
