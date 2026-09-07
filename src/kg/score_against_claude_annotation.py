"""
score_against_claude_annotation.py

Scores the three models (Qwen3-30B, gpt-oss-20b, mistral-medium-3.5) against
Claude's independent annotation (data/interim/annotation_sample_WITH_CLAUDE.csv)
as a working gold standard, joined against model predictions in
data/interim/annotation_reference_HIDDEN.jsonl (keyed by record_id).

NOTE: This is Claude's annotation, not the user's independent human annotation.
Treat these results as a first-pass / provisional evaluation. Once the user's
own independent annotation is complete, re-run scoring against that as the
primary gold standard, and use this script's output for inter-annotator
comparison instead.

Computes, per model, per relation field (manufactures, distributed_by,
made_in, is_component_of):
    - Precision, Recall, F1 (positive-class only -- i.e. excluding
      "unknown"/"no" trivial-negative agreement, following the same
      positive-macro-F1 principle used in Bhavana Raju's thesis for
      handling heavily imbalanced no_relation-dominated data)
    - A confusion breakdown: true positive / false positive / false negative
      / true negative counts, so you can see exactly where each model over-
      or under-extracts relative to the gold annotation.

For 'manufactures' specifically (binary yes/no), scoring compares the
model's "yes"/"unknown" prediction against Claude's gold "yes"/"no"/"unknown"
judgment. For distributed_by/made_in/is_component_of (string-valued),
scoring uses normalized string comparison: a prediction counts as correct
only if BOTH the model and gold agree there IS a value (non-unknown) AND
the values match after light normalization (lowercase, whitespace-stripped).
This is intentionally strict -- partial/fuzzy string matches are reported
separately as a softer "any signal, correct direction" metric.

Usage:
    python3 src/kg/score_against_claude_annotation.py
"""

import json
import csv

ANNOTATION_CSV = "data/interim/claude_annotations_slim.csv"
REFERENCE_JSONL = "data/interim/annotation_reference_HIDDEN.jsonl"

RELATION_FIELDS = ["manufactures", "distributed_by", "made_in", "is_component_of"]
MODELS = ["qwen", "gptoss20b", "mistral", "llama70b"]


COUNTRY_ALIASES = {
    'holland': 'netherlands',
    'usa': 'united states',
    'u.s.a.': 'united states',
    'u.s.': 'united states',
    'america': 'united states',
    'uk': 'united kingdom',
    'u.k.': 'united kingdom',
    'england': 'united kingdom',
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
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            rid = row.get("record_id", "").strip()
            if not rid or rid == "le Op":  # skip known malformed row
                continue
            gold[rid] = {
                "manufactures": normalize(row.get("claude_manufactures_yes_no")),
                "distributed_by": normalize(row.get("claude_distributed_by")),
                "made_in": normalize(row.get("claude_made_in")),
                "is_component_of": normalize(row.get("claude_is_component_of")),
            }
    return gold


def load_reference(path):
    ref = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            ref[entry["record_id"]] = entry
    return ref


def get_model_prediction(entry, model_key, field):
    preds = entry.get(model_key)
    if preds is None:
        return None  # model has no prediction for this record at all
    return normalize(preds.get(field))


def score_field(gold, reference, model_key, field):
    """Returns dict with tp, fp, fn, tn, precision, recall, f1, and a list
    of disagreement examples for manual inspection."""
    tp = fp = fn = tn = 0
    disagreements = []
    n_no_prediction = 0

    for rid, gold_rec in gold.items():
        entry = reference.get(rid)
        if entry is None:
            continue
        pred = get_model_prediction(entry, model_key, field)
        if pred is None:
            n_no_prediction += 1
            continue

        gold_val = gold_rec[field]

        if field == "manufactures":
            # binary: gold is "yes"/"no"/"unknown", pred is "yes"/"unknown"
            gold_positive = (gold_val == "yes")
            pred_positive = (pred == "yes")
        else:
            gold_positive = (gold_val != "unknown")
            pred_positive = (pred != "unknown")

        if pred_positive and gold_positive:
            if field == "manufactures":
                tp += 1
            elif pred == gold_val or pred in gold_val or gold_val in pred:
                tp += 1
            else:
                fp += 1
                fn += 1
                disagreements.append((rid, gold_val, pred, "value_mismatch"))
        elif pred_positive and not gold_positive:
            fp += 1
            disagreements.append((rid, gold_val, pred, "false_positive"))
        elif not pred_positive and gold_positive:
            fn += 1
            disagreements.append((rid, gold_val, pred, "false_negative"))
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_no_prediction": n_no_prediction,
        "precision": precision, "recall": recall, "f1": f1,
        "disagreements": disagreements,
    }


def main():
    print("[INFO] Loading Claude's annotation (gold)...")
    gold = load_gold(ANNOTATION_CSV)
    print(f"[INFO] Loaded {len(gold)} gold-annotated records")

    print("[INFO] Loading model prediction reference...")
    reference = load_reference(REFERENCE_JSONL)
    print(f"[INFO] Loaded {len(reference)} reference records")

    for model_key in MODELS:
        print("\n" + "=" * 70)
        print(f"MODEL: {model_key}")
        print("=" * 70)
        for field in RELATION_FIELDS:
            result = score_field(gold, reference, model_key, field)
            n_scored = result["tp"] + result["fp"] + result["fn"] + result["tn"]
            print(f"\n-- {field} -- (n={n_scored}, "
                  f"{result['n_no_prediction']} records had no {model_key} prediction)")
            print(f"  TP={result['tp']}  FP={result['fp']}  "
                  f"FN={result['fn']}  TN={result['tn']}")
            print(f"  Precision={result['precision']:.3f}  "
                  f"Recall={result['recall']:.3f}  F1={result['f1']:.3f}")
            if result["disagreements"]:
                print(f"  First 5 disagreements (record_id, gold, {model_key}_pred, type):")
                for d in result["disagreements"][:5]:
                    print(f"    {d}")

    print("\n" + "=" * 70)
    print("SUMMARY TABLE (F1 per model per field)")
    print("=" * 70)
    header = f"{'Field':<18}" + "".join(f"{m:<14}" for m in MODELS)
    print(header)
    for field in RELATION_FIELDS:
        row_str = f"{field:<18}"
        for model_key in MODELS:
            result = score_field(gold, reference, model_key, field)
            row_str += f"{result['f1']:<14.3f}"
        print(row_str)

    print("\n[NOTE] This scoring uses Claude's annotation as a provisional gold "
          "standard. Once your own independent annotation is complete, re-run "
          "scoring against your gold_* columns instead (a modified version of "
          "load_gold() reading gold_manufactures_yes_no etc.), and use this "
          "script's results for inter-annotator comparison.")


if __name__ == "__main__":
    main()
