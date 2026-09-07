"""
build_gold118_scoring.py

Joins the user's independent gold annotation (117 topup_* records) with
Qwen's full-corpus extraction predictions, matched via domain, then scores
precision/recall/F1 for made_in and is_component_of (the two fields covered
by this gold file).

Usage:
    python3 build_gold118_scoring.py \
        data/interim/gold_118_annotation.csv \
        data/interim/annotation_topup_BLIND.csv \
        data/interim/extraction_results_normalized_v3.jsonl
"""

import csv
import json
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
    return COUNTRY_ALIASES.get(s, s)


def load_gold(path):
    gold = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gold[row["record_id"]] = {
                "made_in": normalize(row["gold_made_in"]),
                "is_component_of": normalize(row["gold_is_component_of"]),
            }
    return gold


def load_topup_domain_map(path):
    """Maps topup_XXXX record_id -> domain"""
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


def load_qwen_predictions(path):
    """Maps domain -> {made_in, is_component_of} from Qwen's full-corpus results"""
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
                    "made_in": normalize(rec.get("made_in", "unknown")),
                    "is_component_of": normalize(rec.get("is_component_of", "unknown")),
                }
    return preds


def score_field(gold, qwen_preds, domain_map, field):
    tp = fp = fn = tn = 0
    n_no_domain = 0
    n_no_pred = 0
    disagreements = []

    for rid, gold_vals in gold.items():
        domain = domain_map.get(rid)
        if not domain:
            n_no_domain += 1
            continue
        pred_entry = qwen_preds.get(domain)
        if pred_entry is None:
            n_no_pred += 1
            continue

        gold_val = gold_vals[field]
        pred_val = pred_entry[field]

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

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_no_domain": n_no_domain, "n_no_pred": n_no_pred,
        "precision": precision, "recall": recall, "f1": f1,
        "disagreements": disagreements,
    }


def main():
    if len(sys.argv) != 4:
        print("Usage: python3 build_gold118_scoring.py <gold_csv> <topup_blind_csv> <qwen_full_jsonl>")
        sys.exit(1)

    gold_path, topup_blind_path, qwen_full_path = sys.argv[1], sys.argv[2], sys.argv[3]

    print("[INFO] Loading gold annotation...")
    gold = load_gold(gold_path)
    print(f"[INFO] Loaded {len(gold)} gold records")

    print("[INFO] Loading topup record_id -> domain mapping...")
    domain_map = load_topup_domain_map(topup_blind_path)
    print(f"[INFO] Loaded {len(domain_map)} domain mappings")

    print("[INFO] Loading Qwen full-corpus predictions...")
    qwen_preds = load_qwen_predictions(qwen_full_path)
    print(f"[INFO] Loaded {len(qwen_preds)} Qwen predictions")

    for field in ["made_in", "is_component_of"]:
        result = score_field(gold, qwen_preds, domain_map, field)
        n_scored = result["tp"] + result["fp"] + result["fn"] + result["tn"]
        print(f"\n{'='*60}")
        print(f"FIELD: {field}")
        print(f"{'='*60}")
        print(f"  Scored: {n_scored} | No domain match: {result['n_no_domain']} | No Qwen pred: {result['n_no_pred']}")
        print(f"  TP={result['tp']}  FP={result['fp']}  FN={result['fn']}  TN={result['tn']}")
        print(f"  Precision={result['precision']:.3f}  Recall={result['recall']:.3f}  F1={result['f1']:.3f}")
        if result["disagreements"]:
            print(f"  First 5 disagreements (record_id, gold, qwen_pred, type):")
            for d in result["disagreements"][:5]:
                print(f"    {d}")


if __name__ == "__main__":
    main()
