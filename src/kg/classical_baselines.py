"""
classical_baselines.py

Classical ML baselines for supply chain relation extraction, following the
methodological paradigm of Kim et al. (2025), who compare SVM (bag-of-words),
MLP, and Bi-LSTM against BERT/RoBERTa/Llama for supply chain relation
classification. This thesis substitutes TF-IDF-weighted bag-of-words
features (a standard extension of plain bag-of-words) with two classifiers
directly following that paradigm: SVM and Logistic Regression.

Wichmann et al. (2020)'s original CNN-based method and training corpus were
not accessible for direct reimplementation; this classical-ML comparison
follows the precedent set by Kim et al. (2025), who used SVM as one of
their own baseline models in the same research area, rather than
attempting to reproduce Wichmann's specific architecture.

SCOPE:
- manufactures: 3-class classification (yes/no/unknown), naturally categorical.
- distributed_by, made_in, is_component_of: binary classification only
  (unknown vs. "named"), since SVM/LogReg are fixed-label classifiers and
  cannot perform open-ended entity extraction.

Training data: 366-row primary sample + 800-row top-up sample (Claude-
annotated), MINUS any record overlapping with the 117-row gold test set
(avoids data leakage). NOTE: this means the baseline is trained on
Claude-annotated labels but evaluated against the user's independently
verified gold standard -- a stricter test than the training labels
themselves. This asymmetry is documented, not concealed.

Usage:
    python3 classical_baselines.py \
        data/interim/annotation_sample_WITH_CLAUDE.csv \
        data/interim/annotation_topup_WITH_CLAUDE.csv \
        data/interim/gold_118_annotation.csv
"""

import csv
import re
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def normalize(s):
    if not s:
        return "unknown"
    s = str(s).strip().lower()
    cleaned = re.sub(r'[(),]', ' ', s)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned if cleaned else "unknown"


def binarize(s):
    return "unknown" if normalize(s) == "unknown" else "named"


def load_csv_records(path, claude_prefix="claude_"):
    records = {}
    with open(path, newline="", encoding="utf-8") as f:
        sample = f.read(2048)
        f.seek(0)
        delimiter = ";" if sample.count(";") > sample.count(",") else ","
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            rid = row.get("record_id", "").strip()
            if not rid:
                continue
            text = row.get("archived_page_text", "")
            records[rid] = {
                "text": text,
                "manufactures": normalize(row.get(f"{claude_prefix}manufactures_yes_no", "unknown")),
                "distributed_by": normalize(row.get(f"{claude_prefix}distributed_by", "unknown")),
                "made_in": normalize(row.get(f"{claude_prefix}made_in", "unknown")),
                "is_component_of": normalize(row.get(f"{claude_prefix}is_component_of", "unknown")),
            }
    return records


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


def load_gold_texts(candidates, gold_ids):
    texts = {}
    for path in candidates:
        try:
            with open(path, newline="", encoding="utf-8") as f:
                sample = f.read(2048)
                f.seek(0)
                delimiter = ";" if sample.count(";") > sample.count(",") else ","
                reader = csv.DictReader(f, delimiter=delimiter)
                for row in reader:
                    rid = row.get("record_id", "").strip()
                    if rid in gold_ids and rid not in texts:
                        texts[rid] = row.get("archived_page_text", "")
        except FileNotFoundError:
            continue
    return texts


def train_and_eval_field(field, clf_name, clf, train_records, gold_records, gold_texts, binary=False):
    train_ids = list(train_records.keys())
    X_train_text = [train_records[rid]["text"] for rid in train_ids]
    if binary:
        y_train = [binarize(train_records[rid][field]) for rid in train_ids]
    else:
        y_train = [train_records[rid][field] for rid in train_ids]

    test_ids = [rid for rid in gold_records if rid in gold_texts]
    X_test_text = [gold_texts[rid] for rid in test_ids]
    if binary:
        y_test = [binarize(gold_records[rid][field]) for rid in test_ids]
    else:
        y_test = [gold_records[rid][field] for rid in test_ids]

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    return {
        "clf_name": clf_name,
        "n_train": len(train_ids), "n_test": len(test_ids),
        "accuracy": acc, "precision": precision, "recall": recall, "f1": f1,
    }


def main():
    if len(sys.argv) != 4:
        print("Usage: python3 classical_baselines.py <primary_csv> <topup_csv> <gold_csv>")
        sys.exit(1)

    primary_path, topup_path, gold_path = sys.argv[1], sys.argv[2], sys.argv[3]

    print("[INFO] Loading training data (primary + topup)...")
    primary_records = load_csv_records(primary_path)
    topup_records = load_csv_records(topup_path)
    all_train_candidates = {**primary_records, **topup_records}
    print(f"[INFO] {len(all_train_candidates)} candidate training records loaded")

    print("[INFO] Loading gold test set...")
    gold_records = load_gold(gold_path)
    gold_ids = set(gold_records.keys())
    print(f"[INFO] {len(gold_ids)} gold test records")

    train_records = {rid: rec for rid, rec in all_train_candidates.items() if rid not in gold_ids}
    n_removed = len(all_train_candidates) - len(train_records)
    print(f"[INFO] Removed {n_removed} overlapping records from training set (avoiding leakage)")
    print(f"[INFO] Final training set: {len(train_records)} records")

    print("[INFO] Loading page text for gold test records...")
    gold_texts = load_gold_texts([primary_path, topup_path], gold_ids)
    print(f"[INFO] Found text for {len(gold_texts)} / {len(gold_ids)} gold records")

    print("\n" + "="*90)
    print("CLASSICAL ML BASELINES: SVM and Logistic Regression (TF-IDF features)")
    print("Following the classical-baseline paradigm of Kim et al. (2025)")
    print("="*90)

    fields_config = [
        ("manufactures", False),
        ("distributed_by", True),
        ("made_in", True),
        ("is_component_of", True),
    ]

    classifiers = [
        ("SVM (LinearSVC)", LinearSVC(class_weight="balanced", max_iter=5000)),
        ("Logistic Regression", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ]

    for field, binary in fields_config:
        label = f"{field} (3-class: yes/no/unknown)" if not binary else f"{field} (binary: unknown vs. named)"
        print(f"\n-- {label} --")
        for clf_name, clf in classifiers:
            result = train_and_eval_field(field, clf_name, clf, train_records, gold_records, gold_texts, binary=binary)
            print(f"  {clf_name:<22} Train n={result['n_train']}, Test n={result['n_test']}  "
                  f"Accuracy={result['accuracy']:.3f}  Macro-F1={result['f1']:.3f}  "
                  f"(P={result['precision']:.3f} R={result['recall']:.3f})")


if __name__ == "__main__":
    main()
