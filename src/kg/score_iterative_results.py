"""
score_iterative_results.py

Scores the iterative extraction pipeline's results (5-call, per-relation
decomposition) against the user's 117-row independent gold standard,
using the exact same scoring logic as build_gold118_scoring.py for a
direct, apples-to-apples comparison against the single-pass baseline.

Usage:
    python3 score_iterative_results.py \
        data/interim/gold_118_annotation.csv \
        data/interim/iterative_results_117.jsonl
"""

import csv
import json
import sys
import re


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


def load_iterative_predictions(path):
    """Maps domain -> predictions from the iterative pipeline output"""
    preds = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            domain = rec.get("domain", "").strip()
            if domain:
                preds[domain] = {
                    "manufactures": normalize(rec.get("manufactures", "unknown")),
                    "distributed_by": normalize(rec.get("distributed_by", "unknown")),
                    "made_in": normalize(rec.get("made_in", "unknown")),
                    "is_component_of": normalize(rec.get("is_component_of", "unknown")),
                }
    return preds


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


def score_field(gold, preds, domain_map, field):
    tp = fp = fn = tn = 0
    disagreements = []

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
                disagreements.append((rid, gold_val, pred_val, "false_positive"))
            elif not pred_positive and gold_positive:
                fn += 1
                disagreements.append((rid, gold_val, pred_val, "false_negative"))
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
                disagreements.append((rid, gold_val, pred_val, "value_mismatch"))
        elif pred_positive and not gold_positive:
            fp += 1
            disagreements.append((rid, gold_val, pred_val, "false_positive"))
        elif not pred_positive and gold_positive:
            fn += 1
            disagreements.append((rid, gold_val, pred_val, "false_negative"))
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1,
            "disagreements": disagreements}


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 score_iterative_results.py <gold_csv> <iterative_results_jsonl>")
        sys.exit(1)

    gold_path, iterative_path = sys.argv[1], sys.argv[2]
    domain_map_path = "data/interim/annotation_topup_BLIND.csv"

    print("[INFO] Loading gold annotation...")
    gold = load_gold(gold_path)
    print(f"[INFO] Loaded {len(gold)} gold records")

    print("[INFO] Loading topup domain mapping...")
    domain_map = load_topup_domain_map(domain_map_path)

    print("[INFO] Loading iterative extraction predictions...")
    preds = load_iterative_predictions(iterative_path)
    print(f"[INFO] Loaded {len(preds)} predictions")

    print("\n" + "="*70)
    print("ITERATIVE EXTRACTION RESULTS (5-call per-relation pipeline)")
    print("="*70)

    for field in ["manufactures", "distributed_by", "made_in", "is_component_of"]:
        result = score_field(gold, preds, domain_map, field)
        n_scored = result["tp"] + result["fp"] + result["fn"] + result["tn"]
        print(f"\n-- {field} -- (n={n_scored})")
        print(f"  TP={result['tp']}  FP={result['fp']}  FN={result['fn']}  TN={result['tn']}")
        print(f"  Precision={result['precision']:.3f}  Recall={result['recall']:.3f}  F1={result['f1']:.3f}")
        if result["disagreements"]:
            print(f"  First 5 disagreements:")
            for d in result["disagreements"][:5]:
                print(f"    {d}")


if __name__ == "__main__":
    main()
