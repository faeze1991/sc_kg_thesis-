"""
Stage D - Entity normalization for supply chain KG construction.

Handles:
  - company / distributed_by: strip legal suffixes, detect domain-fallback
    values, fuzzy-merge genuine company names
  - product / is_component_of: light string cleanup only (no merging)
  - made_in: alias-map based normalization (see location_aliases.py)

Run on a SAMPLE first before the full corpus - see main() sample_size arg.
"""

import json
import re
import sys
from collections import defaultdict
from rapidfuzz import fuzz

from location_aliases import normalize_location

LEGAL_SUFFIXES = [
    r"\bltd\.?\b", r"\bllc\.?\b", r"\binc\.?\b", r"\bcorp\.?\b",
    r"\bcorporation\b", r"\bgmbh\b", r"\bco\.?\b", r"\bcompany\b",
    r"\bs\.a\.?\b", r"\bb\.v\.?\b", r"\bsas\b", r"\bkg\b", r"\bag\b",
    r"\bplc\b", r"\bpvt\.?\b", r"\bpty\.?\b",
]
SUFFIX_PATTERN = re.compile("|".join(LEGAL_SUFFIXES), flags=re.IGNORECASE)

TRADEMARK_CHARS = str.maketrans("", "", "®™©")

FUZZY_MATCH_THRESHOLD = 97  # conservative - prefer under-merging over false merges


def clean_string(raw: str) -> str:
    """Basic cleanup: trademark symbols, legal suffixes, whitespace/punct normalization."""
    if not raw:
        return raw
    s = raw.translate(TRADEMARK_CHARS)
    s = SUFFIX_PATTERN.sub("", s)
    s = re.sub(r"[,\.]+$", "", s.strip())  # trailing commas/periods
    s = re.sub(r"\s+", " ", s).strip()
    return s


def domain_root(domain: str) -> str:
    """Strip www. and TLD-ish suffix for comparing company name against domain."""
    d = domain.lower()
    d = re.sub(r"^www\.", "", d)
    d = re.split(r"\.", d)[0]  # take first label, e.g. 'skywayparts' from 'skywayparts.ca'
    return d


def is_domain_fallback(company_value: str, domain: str) -> bool:
    """
    Heuristic: is the 'company' value just the domain (or its root) restated,
    rather than a genuine extracted company name?
    """
    if not company_value:
        return False
    cv = company_value.lower().strip()
    cv_alnum = re.sub(r"[^a-z0-9]", "", cv)
    droot = domain_root(domain)
    droot_alnum = re.sub(r"[^a-z0-9]", "", droot)
    full_domain_alnum = re.sub(r"[^a-z0-9]", "", domain.lower())
    return cv_alnum == droot_alnum or cv_alnum == full_domain_alnum


def build_fuzzy_clusters(names: list, threshold: int = FUZZY_MATCH_THRESHOLD) -> dict:
    """
    Greedy clustering with blocking by first letter: for each unique cleaned
    name, only compare against existing cluster representatives that share the
    same first character (case-insensitive). This narrows the O(n * clusters)
    comparison space substantially since true near-duplicates virtually always
    share a first letter (fuzzy matching handles the rest of the string).
    Returns: {original_name: canonical_name}
    """
    blocks = defaultdict(list)  # first-char key -> list of (representative, [members])
    mapping = {}

    for name in names:
        key = name[:1].lower() if name else ""
        cluster_list = blocks[key]
        matched = False
        for rep, members in cluster_list:
            score = fuzz.token_sort_ratio(name.lower(), rep.lower())
            if score >= threshold:
                members.append(name)
                matched = True
                break
        if not matched:
            cluster_list.append((name, [name]))

    for cluster_list in blocks.values():
        for rep, members in cluster_list:
            for m in members:
                mapping[m] = rep

    return mapping


def normalize_record(record: dict) -> dict:
    """Apply light cleanup to product / is_component_of fields. Company-level
    fuzzy merging is applied separately as a corpus-wide pass (see main)."""
    out = dict(record)

    for field in ("product", "is_component_of"):
        val = out.get(field, "unknown")
        if val and val.lower() != "unknown":
            out[field] = clean_string(val)

    made_in = out.get("made_in", "unknown")
    if made_in and made_in.lower() != "unknown":
        out["made_in"] = normalize_location(made_in)

    return out


def main(input_path, output_path, sample_size=None):
    records = []
    with open(input_path) as f:
        for i, line in enumerate(f):
            if sample_size and i >= sample_size:
                break
            records.append(json.loads(line))

    print(f"Loaded {len(records):,} records")

    # Pass 1: clean company/distributed_by strings, detect domain fallback
    genuine_companies = set()
    genuine_distributors = set()

    for r in records:
        company = r.get("company", "unknown")
        if company and company.lower() != "unknown":
            cleaned = clean_string(company)
            r["company_cleaned"] = cleaned
            r["company_is_domain_fallback"] = is_domain_fallback(cleaned, r["domain"])
            if not r["company_is_domain_fallback"]:
                genuine_companies.add(cleaned)

        dist = r.get("distributed_by", "unknown")
        if dist and dist.lower() != "unknown":
            cleaned = clean_string(dist)
            r["distributed_by_cleaned"] = cleaned
            genuine_distributors.add(cleaned)

    print(f"Genuine (non-domain-fallback) company names: {len(genuine_companies):,}")
    print(f"Genuine distributed_by names: {len(genuine_distributors):,}")

    # Pass 2: fuzzy-cluster genuine names
    print("Clustering company names...")
    company_map = build_fuzzy_clusters(sorted(genuine_companies))
    print("Clustering distributor names...")
    distributor_map = build_fuzzy_clusters(sorted(genuine_distributors))

    # Pass 3: apply canonical names + light cleanup for product/location fields
    out_records = []
    for r in records:
        r = normalize_record(r)
        if r.get("company_cleaned") and not r.get("company_is_domain_fallback"):
            r["company_canonical"] = company_map.get(r["company_cleaned"], r["company_cleaned"])
        if r.get("distributed_by_cleaned"):
            r["distributed_by_canonical"] = distributor_map.get(
                r["distributed_by_cleaned"], r["distributed_by_cleaned"]
            )
        out_records.append(r)

    with open(output_path, "w") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(out_records):,} records to {output_path}")


if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else "data/interim/extraction_results_full_qwen.jsonl"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/interim/extraction_results_normalized.jsonl"
    sample_size = int(sys.argv[3]) if len(sys.argv) > 3 else None
    main(input_path, output_path, sample_size)
