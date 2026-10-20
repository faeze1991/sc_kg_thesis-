"""
build_annotation_sample.py

HISTORICAL (earlier annotation round): this script drew the pages of the first
annotation round. Those pages were later labelled with LLM assistance and are
used only as "silver" training data for the classical baseline
(data/annotation/silver/). They are NOT part of the gold-standard evaluation,
which uses the 320 stratified pages (make_stratified_annotation_sheet.py).

Builds TWO outputs for manual annotation / evaluation:

1. annotation_sample_BLIND.csv
   -- What you actually fill in. Contains ONLY domain/url/product/company
      context (no model predictions) plus empty gold_* columns for you to
      fill in based on the real page content. Kept blind on purpose: seeing
      a model's prediction while annotating biases your judgment toward it.

2. annotation_reference_HIDDEN.jsonl
   -- NOT for you to open while annotating. Contains each record's
      predictions from all available models (Qwen / gpt-oss-20b / mistral),
      keyed by record_id, for scoring against your gold labels afterward.

Sampling strategy:
    - ALL records where Qwen + gpt-oss-20b + mistral all have results
      (the ~300-page overlap subset) -> enables 3-way model comparison.
    - +100 additional records from the remaining Qwen+gpt-oss-only pool,
      stratified to match the known ~45%/55% signal/no-signal split from
      the full-corpus run, so the sample isn't skewed toward only the
      "interesting" (signal) records.

Usage:
    python -m src.kg.build_annotation_sample
"""

import json
import random
import csv

random.seed(42)  # reproducible sample -- document this seed in your methodology section

BENCH2K_PATH = "data/interim/fetched_pages_bench2k.jsonl"
MISTRAL300_PATH = "data/interim/fetched_pages_mistral300.jsonl"
QWEN_NORMALIZED_PATH = "data/interim/extraction_results_normalized_FINAL.jsonl"
GPTOSS_PATH = "data/interim/extraction_results_gptoss20b_bench2k.jsonl"
MISTRAL_PATH = "data/interim/extraction_results_mistral_300.jsonl"

OUT_BLIND_CSV = "data/interim/annotation_sample_BLIND.csv"
OUT_REFERENCE_JSONL = "data/interim/annotation_reference_HIDDEN.jsonl"

N_ADDITIONAL_FROM_REMAINDER = 100

RELATION_FIELDS = ["manufactures", "distributed_by", "made_in", "is_component_of"]


def load_jsonl_by_domain(path):
    records = {}
    try:
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                domain = rec.get("domain")
                if domain:
                    records[domain] = rec
    except FileNotFoundError:
        print(f"[WARN] {path} not found -- treating as empty (0 records).")
    return records


def has_signal(rec):
    return any(rec.get(f, "unknown") != "unknown" for f in RELATION_FIELDS)


def extract_predictions(rec):
    if rec is None:
        return None
    return {f: rec.get(f, "unknown") for f in RELATION_FIELDS}


def main():
    mistral300_domains = set(load_jsonl_by_domain(MISTRAL300_PATH).keys())
    bench2k_domains = set(load_jsonl_by_domain(BENCH2K_PATH).keys())

    qwen_by_domain = load_jsonl_by_domain(QWEN_NORMALIZED_PATH)
    gptoss_by_domain = load_jsonl_by_domain(GPTOSS_PATH)
    mistral_by_domain = load_jsonl_by_domain(MISTRAL_PATH)

    # Restrict Qwen lookup to just the bench2k domains (Qwen's normalized file
    # covers the full 280K corpus -- we only want the benchmark subset here)
    qwen_bench = {d: r for d, r in qwen_by_domain.items() if d in bench2k_domains}

    print(f"[INFO] bench2k domains: {len(bench2k_domains)}")
    print(f"[INFO] mistral300 domains: {len(mistral300_domains)}")
    print(f"[INFO] Qwen records in bench2k: {len(qwen_bench)}")
    print(f"[INFO] gpt-oss records in bench2k: {len(gptoss_by_domain)}")
    print(f"[INFO] mistral records available: {len(mistral_by_domain)}")

    # --- Tier 1: full 3-way overlap (mistral300 domains that all 3 models cover) ---
    overlap_domains = [
        d for d in mistral300_domains
        if d in qwen_bench and d in gptoss_by_domain and d in mistral_by_domain
    ]
    print(f"[INFO] Full 3-way overlap (Qwen+gptoss+mistral): {len(overlap_domains)} domains")

    # --- Tier 2: remainder pool (Qwen+gptoss only, excluding tier 1) ---
    remainder_domains = [
        d for d in bench2k_domains
        if d in qwen_bench and d in gptoss_by_domain and d not in overlap_domains
    ]
    signal_pool = [d for d in remainder_domains if has_signal(qwen_bench[d])]
    nosignal_pool = [d for d in remainder_domains if not has_signal(qwen_bench[d])]

    n_signal = round(N_ADDITIONAL_FROM_REMAINDER * 0.45)
    n_nosignal = N_ADDITIONAL_FROM_REMAINDER - n_signal
    n_signal = min(n_signal, len(signal_pool))
    n_nosignal = min(n_nosignal, len(nosignal_pool))

    sampled_remainder = (
        random.sample(signal_pool, n_signal) +
        random.sample(nosignal_pool, n_nosignal)
    )
    print(f"[INFO] Additional stratified sample from remainder: "
          f"{n_signal} signal + {n_nosignal} no-signal = {len(sampled_remainder)}")

    all_selected_domains = overlap_domains + sampled_remainder
    random.shuffle(all_selected_domains)  # so tier 1/2 aren't grouped in the CSV
    print(f"[INFO] TOTAL sample size: {len(all_selected_domains)}")

    # --- Write blind annotation CSV ---
    with open(OUT_BLIND_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "record_id", "domain", "url", "product", "company",
            "gold_manufactures_yes_no", "gold_distributed_by",
            "gold_made_in", "gold_is_component_of", "annotator_notes"
        ])
        for i, domain in enumerate(all_selected_domains):
            rec = qwen_bench[domain]
            record_id = f"ann_{i:04d}"
            writer.writerow([
                record_id, domain, rec.get("url", ""),
                rec.get("product", ""), rec.get("company", ""),
                "", "", "", "", ""
            ])
    print(f"[INFO] Wrote blind annotation CSV: {OUT_BLIND_CSV}")

    # --- Write hidden reference JSONL (predictions from all models) ---
    with open(OUT_REFERENCE_JSONL, "w", encoding="utf-8") as f:
        for i, domain in enumerate(all_selected_domains):
            record_id = f"ann_{i:04d}"
            entry = {
                "record_id": record_id,
                "domain": domain,
                "in_3way_overlap": domain in overlap_domains,
                "qwen": extract_predictions(qwen_bench.get(domain)),
                "gptoss20b": extract_predictions(gptoss_by_domain.get(domain)),
                "mistral": extract_predictions(mistral_by_domain.get(domain)),
            }
            f.write(json.dumps(entry) + "\n")
    print(f"[INFO] Wrote hidden reference file: {OUT_REFERENCE_JSONL}")
    print("\n[REMINDER] Do NOT open annotation_reference_HIDDEN.jsonl while annotating -- "
          "it contains model predictions and will bias your judgments.")


if __name__ == "__main__":
    main()
