"""compare_models.py - corrected: manufactures checks for 'yes' specifically, not just non-unknown."""

import json
from collections import Counter

BENCH2K_PATH = "data/interim/fetched_pages_bench2k.jsonl"
MISTRAL300_PATH = "data/interim/fetched_pages_mistral300.jsonl"
QWEN_NORMALIZED_PATH = "data/interim/extraction_results_normalized_FINAL.jsonl"
GPTOSS_PATH = "data/interim/extraction_results_gptoss20b_bench2k.jsonl"
MISTRAL_PATH = "data/interim/extraction_results_mistral_300.jsonl"

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


def normalize_value(v):
    if v is None:
        return "unknown"
    v = str(v).strip().lower()
    return v if v else "unknown"


def is_positive(rec, field):
    """CORRECTED: manufactures is binary (yes/no/unknown) -- only 'yes' is
    positive signal. Other fields are string-valued -- any non-'unknown'
    value is positive signal."""
    v = normalize_value(rec.get(field))
    if field == "manufactures":
        return v == "yes"
    return v != "unknown"


def has_signal(rec):
    return any(is_positive(rec, f) for f in RELATION_FIELDS)


def signal_stats(records, label):
    n = len(records)
    if n == 0:
        print(f"\n=== {label}: no records ===")
        return
    n_signal = sum(1 for r in records.values() if has_signal(r))
    print(f"\n=== {label} (n={n}) ===")
    print(f"Signal rate (>=1 positive relation): {n_signal}/{n} ({100*n_signal/n:.1f}%)")
    for field in RELATION_FIELDS:
        n_pos = sum(1 for r in records.values() if is_positive(r, field))
        print(f"  {field}: {n_pos}/{n} ({100*n_pos/n:.1f}%)")
    # explicit breakdown for manufactures since it's the field with the fix
    m_values = Counter(normalize_value(r.get("manufactures")) for r in records.values())
    print(f"  [manufactures full breakdown] {dict(m_values)}")


def pairwise_agreement(records_a, records_b, label_a, label_b):
    common_domains = set(records_a.keys()) & set(records_b.keys())
    print(f"\n=== Agreement: {label_a} vs {label_b} (n={len(common_domains)} shared domains) ===")

    for field in RELATION_FIELDS:
        both_negative = 0
        substantive_agree = 0
        disagree = 0
        for d in common_domains:
            va = normalize_value(records_a[d].get(field))
            vb = normalize_value(records_b[d].get(field))
            pos_a = is_positive(records_a[d], field)
            pos_b = is_positive(records_b[d], field)

            if not pos_a and not pos_b:
                both_negative += 1
            elif pos_a and pos_b and va == vb:
                substantive_agree += 1
            else:
                disagree += 1
        total = len(common_domains)
        print(f"  {field}:")
        print(f"    both negative (trivial agreement): {both_negative}/{total} ({100*both_negative/total:.1f}%)")
        print(f"    same positive value (substantive agreement): {substantive_agree}/{total} ({100*substantive_agree/total:.1f}%)")
        print(f"    disagree: {disagree}/{total} ({100*disagree/total:.1f}%)")

    both_signal_agree = sum(
        1 for d in common_domains
        if has_signal(records_a[d]) == has_signal(records_b[d])
    )
    print(f"  [overall] agree on whether record has ANY positive signal: "
          f"{both_signal_agree}/{len(common_domains)} "
          f"({100*both_signal_agree/len(common_domains):.1f}%)")


def main():
    bench2k_domains = set(load_jsonl_by_domain(BENCH2K_PATH).keys())
    mistral300_domains = set(load_jsonl_by_domain(MISTRAL300_PATH).keys())

    qwen_all = load_jsonl_by_domain(QWEN_NORMALIZED_PATH)
    qwen_bench = {d: r for d, r in qwen_all.items() if d in bench2k_domains}
    gptoss_bench = load_jsonl_by_domain(GPTOSS_PATH)
    mistral_records = load_jsonl_by_domain(MISTRAL_PATH)

    print("#" * 70)
    print("PER-MODEL SIGNAL / EXTRACTION RATES (CORRECTED)")
    print("#" * 70)
    signal_stats(qwen_bench, "Qwen3-30B (2,000-page bench2k)")
    signal_stats(gptoss_bench, "gpt-oss-20b (2,000-page bench2k)")
    signal_stats(mistral_records, "mistral-medium-3.5 (266/300 completed)")

    print("\n" + "#" * 70)
    print("PAIRWISE AGREEMENT (CORRECTED)")
    print("#" * 70)
    pairwise_agreement(qwen_bench, gptoss_bench, "Qwen", "gpt-oss-20b")

    qwen_overlap = {d: r for d, r in qwen_bench.items() if d in mistral300_domains}
    gptoss_overlap = {d: r for d, r in gptoss_bench.items() if d in mistral300_domains}

    pairwise_agreement(qwen_overlap, mistral_records, "Qwen", "mistral (3-way overlap)")
    pairwise_agreement(gptoss_overlap, mistral_records, "gpt-oss-20b", "mistral (3-way overlap)")


if __name__ == "__main__":
    main()
