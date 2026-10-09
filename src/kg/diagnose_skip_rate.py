"""
diagnose_skip_rate.py

Investigates why Llama's normalized extraction has a ~3x higher "no usable
product string" skip rate than Qwen's (7.4% vs 2.4%) when building the KG.
"""

import json
import sys


def analyze(path, label):
    total = 0
    empty_product = 0
    unknown_product = 0
    unknown_company = 0
    both_unknown = 0
    samples = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            product = str(rec.get("product", "")).strip()
            company = str(rec.get("company", "")).strip()

            is_empty = not product
            is_unknown = product.lower() == "unknown"
            company_unknown = company.lower() == "unknown" or not company

            if is_empty:
                empty_product += 1
            if is_unknown:
                unknown_product += 1
            if company_unknown:
                unknown_company += 1
            if is_unknown and company_unknown:
                both_unknown += 1
                if len(samples) < 10:
                    samples.append({"domain": rec.get("domain"), "company": company, "product": product})

    print(f"\n=== {label} ===")
    print(f"Total records: {total:,}")
    print(f"Empty product field: {empty_product:,} ({100*empty_product/total:.2f}%)")
    print(f"Product == 'unknown': {unknown_product:,} ({100*unknown_product/total:.2f}%)")
    print(f"Company == 'unknown'/empty: {unknown_company:,} ({100*unknown_company/total:.2f}%)")
    print(f"BOTH unknown (likely relevance-check filtered): {both_unknown:,} ({100*both_unknown/total:.2f}%)")
    print(f"Sample records where both are unknown:")
    for s in samples[:5]:
        print(f"  {s}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 diagnose_skip_rate.py <qwen_normalized.jsonl> <llama_normalized.jsonl>")
        sys.exit(1)

    analyze(sys.argv[1], "QWEN")
    analyze(sys.argv[2], "LLAMA")


if __name__ == "__main__":
    main()
