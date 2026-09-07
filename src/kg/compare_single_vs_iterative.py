"""
compare_single_vs_iterative.py

Scores BOTH the single-pass and iterative full-corpus Qwen extraction
results against the user's 117-row independent gold standard, on all 4
relation fields, and prints a direct side-by-side comparison table.
"""

import csv
import json
import re
import sys


COUNTRY_ALIASES = {
    'holland': 'netherlands', 'usa': 'united states', 'u.s.a.': 'united states',
    'u.s.': 'united states', 'america': 'united states', 'uk': 'united kingdom',
    'u.k.': 'united kingdom', 'england': 'united kingdom',
}


def normalize(s):
    if not s:
        return "unknown"
    s = str(s).strip().lower()
    if not s:
        return "unknown"
    cleaned = re.sub(r'[(),]', ' ', s)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return COUNTRY_ALIASES.get(cleaned, cleaned)


def load_gold(path):
    gold = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gold[row["record_id"]] = {
                "manufactures": normalize(row["gold_manufactures_yes_no"]),
                "distributed_by": normalize(row["gold_distributed_by"]),
                "made_in": normalize(row["gold_made_in"]),
                "is_component_of": normalize(row["gold_is_component_of"]),
            }
    return gold


def load_topup_domain_map(path):
    mapping = {}
    with open(path, newline="", encoding="utf-8") as f:
        sample = f.read(2048)
        f.seek(0)
        delimiter = ";" if sample.count(";") > sample.count(",") else ","
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            rid = row.get("record_id", "").strip()
            domain = row.get("domain", "").strip()
            if rid and domain:
                mapping[rid] = domain
    return mapping


def load_predictions_by_domain(path):
    preds = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            domain = rec.get("domain", "").strip()
            if domain:
                preds[domain] = {
                    "manufactures": normalize(rec.get("manufactures", "unknown")),
                    "distributed_by": normalize(rec.get("distributed_by", "unknown")),
                    "made_in": normalize(rec.get("made_in", "unknown")),
                    "is_component_of": normalize(rec.get("is_component_of", "unknown")),
                }
    return preds


def score_field(gold, preds, domain_map, field):
    tp = fp = fn = tn = 0

    for rid, gold_vals in gold.items():
        domain = domain_map.get(rid)
        if not domain:
            continue
        pred_entry = preds.get(domain)
        if pred_entry is None:
            continue

        gold_val = gold_vals[field]
        pred_val = pred_entry[field]

        if field == "manufactures":
            gold_positive = gold_val == "yes"
            pred_positive = pred_val == "yes"
            if pred_positive and gold_positive:
                tp += 1
            elif pred_positive and not gold_positive:
                fp += 1
            elif not pred_positive and gold_positive:
                fn += 1
            else:
                tn += 1
            continue

        gold_positive = gold_val != "unknown"
        pred_positive = pred_val != "unknown"

        if pred_positive and gold_positive:
            if pred_val == gold_val or pred_val in gold_val or gold_val in pred_val:
                tp += 1
            else:
                fp += 1
                fn += 1
        elif pred_positive and not gold_positive:
            fp += 1
        elif not pred_positive and gold_positive:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1}


def main():
    if len(sys.argv) != 5:
        print("Usage: python3 compare_single_vs_iterative.py <gold_csv> <topup_blind_csv> <single_pass_jsonl> <iterative_jsonl>")
        sys.exit(1)

    gold_path, topup_blind_path, single_pass_path, iterative_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

    print("[INFO] Loading gold annotation...")
    gold = load_gold(gold_path)
    print(f"[INFO] Loaded {len(gold)} gold records")

    print("[INFO] Loading topup domain mapping...")
    domain_map = load_topup_domain_map(topup_blind_path)

    print("[INFO] Loading single-pass predictions...")
    single_preds = load_predictions_by_domain(single_pass_path)
    print(f"[INFO] Loaded {len(single_preds)} single-pass predictions")

    print("[INFO] Loading iterative predictions...")
    iter_preds = load_predictions_by_domain(iterative_path)
    print(f"[INFO] Loaded {len(iter_preds)} iterative predictions")

    fields = ["manufactures", "distributed_by", "made_in", "is_component_of"]

    print("\n" + "="*90)
    print("DETAILED RESULTS PER FIELD")
    print("="*90)

    single_results = {}
    iter_results = {}

    for field in fields:
        s = score_field(gold, single_preds, domain_map, field)
        i = score_field(gold, iter_preds, domain_map, field)
        single_results[field] = s
        iter_results[field] = i

        n_scored = s["tp"] + s["fp"] + s["fn"] + s["tn"]
        print(f"\n-- {field} -- (n={n_scored})")
        print(f"  SINGLE-PASS:  TP={s['tp']:>3} FP={s['fp']:>3} FN={s['fn']:>3} TN={s['tn']:>3}  "
              f"P={s['precision']:.3f} R={s['recall']:.3f} F1={s['f1']:.3f}")
        print(f"  ITERATIVE:    TP={i['tp']:>3} FP={i['fp']:>3} FN={i['fn']:>3} TN={i['tn']:>3}  "
              f"P={i['precision']:.3f} R={i['recall']:.3f} F1={i['f1']:.3f}")
        diff = i['f1'] - s['f1']
        sign = "+" if diff >= 0 else ""
        print(f"  F1 change: {sign}{diff:.3f}")

    print("\n" + "="*90)
    print("SUMMARY: F1 SCORE COMPARISON")
    print("="*90)
    print(f"{'Field':<20}{'Single-pass':<15}{'Iterative':<15}{'Change':<10}")
    for field in fields:
        s_f1 = single_results[field]['f1']
        i_f1 = iter_results[field]['f1']
        diff = i_f1 - s_f1
        sign = "+" if diff >= 0 else ""
        print(f"{field:<20}{s_f1:<15.3f}{i_f1:<15.3f}{sign}{diff:.3f}")


if __name__ == "__main__":
    main()
