"""
compare_all_four.py

Scores all four full-corpus extraction variants (Qwen single-pass, Qwen
iterative, Llama single-pass, Llama iterative) against the user's 117-row
independent gold standard, on all 4 relation fields.
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
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1}


def main():
    if len(sys.argv) != 7:
        print("Usage: python3 compare_all_four.py <gold_csv> <topup_blind_csv> "
              "<qwen_single_jsonl> <qwen_iter_jsonl> <llama_single_jsonl> <llama_iter_jsonl>")
        sys.exit(1)

    gold_path, topup_blind_path = sys.argv[1], sys.argv[2]
    model_paths = {
        "Qwen single-pass": sys.argv[3],
        "Qwen iterative": sys.argv[4],
        "Llama single-pass": sys.argv[5],
        "Llama iterative": sys.argv[6],
    }

    print("[INFO] Loading gold annotation...")
    gold = load_gold(gold_path)
    print(f"[INFO] Loaded {len(gold)} gold records")

    print("[INFO] Loading topup domain mapping...")
    domain_map = load_topup_domain_map(topup_blind_path)

    all_preds = {}
    for name, path in model_paths.items():
        print(f"[INFO] Loading {name} predictions from {path}...")
        all_preds[name] = load_predictions_by_domain(path)
        print(f"[INFO]   -> {len(all_preds[name])} records loaded")

    fields = ["manufactures", "distributed_by", "made_in", "is_component_of"]
    all_results = {name: {} for name in model_paths}

    for name in model_paths:
        for field in fields:
            all_results[name][field] = score_field(gold, all_preds[name], domain_map, field)

    print("\n" + "="*100)
    print("FULL COMPARISON: F1 SCORES ACROSS ALL FOUR EXTRACTION VARIANTS")
    print("="*100)
    header = f"{'Field':<20}" + "".join(f"{name:<20}" for name in model_paths)
    print(header)
    for field in fields:
        row = f"{field:<20}"
        for name in model_paths:
            row += f"{all_results[name][field]['f1']:<20.3f}"
        print(row)

    print("\n" + "="*100)
    print("DETAILED PRECISION / RECALL PER FIELD")
    print("="*100)
    for field in fields:
        print(f"\n-- {field} --")
        for name in model_paths:
            r = all_results[name][field]
            n = r['tp'] + r['fp'] + r['fn'] + r['tn']
            print(f"  {name:<20} n={n:<5} TP={r['tp']:>3} FP={r['fp']:>3} FN={r['fn']:>3} TN={r['tn']:>3}  "
                  f"P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f}")


if __name__ == "__main__":
    main()
