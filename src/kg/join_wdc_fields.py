"""
join_wdc_fields.py -- add the WDC fields cluster_id, brand, id and title to each of
our 280,239 selected pages, by exact URL match against the raw WDC corpus.

Why: product identity in the KG needs evidence beyond the extracted name.
  * cluster_id -- WDC groups offers of the same product across websites
                  (check in the WDC documentation how the clusters were built
                  before describing it as identifier-based in the thesis)
  * brand      -- the brand from the site's own schema.org markup (often null):
                  a brand signal independent of the LLM's has_brand
No LLM calls; one streaming pass over data/raw/pdc2020c.json (15 GB).

Usage (from the project root; run in tmux, it takes a while):
    python3 -m src.kg.join_wdc_fields

Output:
    data/interim/wdc_fields.jsonl  one line per matched page:
        {domain, url, wdc_id, wdc_brand, cluster_id, wdc_title}
    plus printed statistics: matches, brand coverage, and how many of OUR pages
    share a cluster_id with at least one other of our pages (cross-site clusters).
"""

import json
from collections import Counter

WDC = "data/raw/pdc2020c.json"
OURS = "data/interim/domain_urls.jsonl"
OUT = "data/interim/wdc_fields.jsonl"


def main():
    want = {}
    with open(OURS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            want[r["url"]] = r["domain"]
    print(f"[INFO] {len(want):,} selected page URLs")

    found = {}
    n = bad = 0
    with open(WDC, encoding="utf-8") as f:
        for line in f:
            n += 1
            if n % 2_000_000 == 0:
                print(f"[INFO] {n:,} WDC lines read, {len(found):,} matches so far", flush=True)
            # cheap pre-check: decode only the url value before parsing the whole line
            i = line.rfind('"url"')
            if i < 0:
                continue
            k = line.find('"', i + 5)          # opening quote of the value
            if k < 0:
                continue
            j = k + 1
            while True:                       # closing quote, skipping escaped ones
                j = line.find('"', j)
                if j < 0:
                    break
                n_bs = 0
                while line[j - 1 - n_bs] == "\\":
                    n_bs += 1
                if n_bs % 2 == 0:
                    break
                j += 1
            if j < 0:
                bad += 1
                continue
            try:
                url = json.loads(line[k:j + 1])
            except json.JSONDecodeError:
                bad += 1
                continue
            if url not in want or url in found:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            found[url] = {"domain": want[url], "url": url, "wdc_id": r.get("id"),
                          "wdc_brand": r.get("brand"), "cluster_id": r.get("cluster_id"),
                          "wdc_title": r.get("title")}

    with open(OUT, "w", encoding="utf-8") as f:
        for rec in found.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_brand = sum(1 for r in found.values() if r["wdc_brand"] not in (None, "", "null"))
    clusters = Counter(r["cluster_id"] for r in found.values() if r["cluster_id"] is not None)
    shared = {c: k for c, k in clusters.items() if k >= 2}
    pages_in_shared = sum(shared.values())
    sizes = Counter(min(k, 10) for k in shared.values())
    print(f"\n[DONE] {OUT}")
    print(f"  WDC lines read: {n:,} ({bad} unparseable)")
    print(f"  matched pages: {len(found):,} of {len(want):,}")
    print(f"  with a WDC brand value: {n_brand:,} ({100 * n_brand / max(len(found), 1):.1f}%)")
    print(f"  distinct cluster_ids among our pages: {len(clusters):,}")
    print(f"  clusters shared by >= 2 of our pages: {len(shared):,} "
          f"(covering {pages_in_shared:,} pages)")
    print(f"  shared-cluster sizes (10 = 10 or more): {dict(sorted(sizes.items()))}")


if __name__ == "__main__":
    main()
