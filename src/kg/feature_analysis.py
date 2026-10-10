"""
feature_analysis.py -- RQ1: what would a semantic pre-filter have gained or lost?
No LLM calls. Uses the existing extraction outputs as the outcome.

For every page, simple pre-extraction features are computed:
  wdc_brand      the WDC record declares a brand (schema.org)
  shared_cluster the page's WDC cluster_id is shared with another of our pages
  long_text      page text >= 3,000 characters
  kw_origin      text (first 3,000 chars) mentions an origin phrase
                 (made in, manufactured in, hergestellt, fabriqué, fabricado, prodotto ...)
  kw_maker       mentions a maker phrase (manufacturer, we make, handmade, hersteller ...)
  kw_compat      mentions a compatibility phrase (compatible with, fits, replacement for,
                 spare part, passend für, ersatzteil ...)
For each feature it reports, per configuration: how many pages have it, the share of
those pages with at least one predicted relation, and -- if a pre-filter had kept
ONLY pages with the feature -- the share of all predicted relations that would have
been retained (per relation type).

NOTE: the outcome is the MODELS' predicted relations, a proxy; the gold sample is too
small for this. Report it as "predicted relation yield", not as accuracy.

Run from the project root:
    python3 -m src.kg.feature_analysis
Output: data/interim/eval/feature_analysis.csv (+ printed table)
"""

import csv
import json
import re
from collections import Counter, defaultdict

D = "data/interim"
CONFIGS = {
    "qwen_single": f"{D}/extraction_results_full_qwen.jsonl",
    "llama_single": f"{D}/extraction_results_llama70b_full.jsonl",
    "qwen_iter": f"{D}/extraction_results_iterative_full.jsonl",
    "llama_iter": f"{D}/extraction_results_iterative_llama_full.jsonl",
}
RELS = {"manufactures": "manufactures", "has_brand": "distributed_by",
        "made_in": "made_in", "is_component_of": "is_component_of"}
NEG = {"", "unknown", "none", "null", "n/a", "na"}
KW = {
    "kw_origin": r"made in|manufactured in|produced in|hergestellt in|fabriqu[ée] en|fabricado en|prodotto in|country of (origin|manufacture)|place of origin",
    "kw_maker": r"manufacturer|we make|we manufacture|handmade|hand made|handcrafted|our (factory|workshop)|hersteller|fabricant|fabricante|produttore",
    "kw_compat": r"compatible with|fits |replacement for|spare part|for use with|passend f[üu]r|ersatzteil|compatible avec|compatibile con",
}


def pos(rel, v):
    v = str(v or "").strip().lower()
    return v == "yes" if rel == "manufactures" else v not in NEG


def main():
    feats = defaultdict(dict)
    with open(f"{D}/wdc_fields.jsonl", encoding="utf-8") as f:
        wdc = [json.loads(l) for l in f]
    cl = Counter(r["cluster_id"] for r in wdc if r.get("cluster_id") is not None)
    for r in wdc:
        feats[r["domain"]]["wdc_brand"] = r.get("wdc_brand") not in (None, "", "null")
        feats[r["domain"]]["shared_cluster"] = cl.get(r.get("cluster_id"), 0) >= 2
    pats = {k: re.compile(p, re.I) for k, p in KW.items()}
    with open(f"{D}/fetched_pages.jsonl", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            d, t = r.get("domain"), r.get("text") or ""
            if not d or d not in feats:
                continue
            feats[d]["long_text"] = len(t) >= 3000
            head = t[:3000]
            for k, p in pats.items():
                feats[d][k] = bool(p.search(head))
    names = ["wdc_brand", "shared_cluster", "long_text", "kw_origin", "kw_maker", "kw_compat"]

    rows = []
    for c, path in CONFIGS.items():
        preds = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                preds.setdefault(r.get("domain"), r)
        doms = [d for d in preds if d in feats]
        total = {rel: sum(pos(rel, preds[d].get(k)) for d in doms) for rel, k in RELS.items()}
        any_all = sum(any(pos(rel, preds[d].get(k)) for rel, k in RELS.items()) for d in doms)
        rows.append({"config": c, "feature": "(all pages)", "pages": len(doms), "share_of_pages": 1.0,
                     "pages_with_any_relation": round(any_all / len(doms), 3),
                     **{f"retained_{rel}": 1.0 for rel in RELS}})
        for fname in names:
            sel = [d for d in doms if feats[d].get(fname)]
            if not sel:
                continue
            anyr = sum(any(pos(rel, preds[d].get(k)) for rel, k in RELS.items()) for d in sel)
            row = {"config": c, "feature": fname, "pages": len(sel),
                   "share_of_pages": round(len(sel) / len(doms), 3),
                   "pages_with_any_relation": round(anyr / len(sel), 3)}
            for rel, k in RELS.items():
                kept = sum(pos(rel, preds[d].get(k)) for d in sel)
                row[f"retained_{rel}"] = round(kept / total[rel], 3) if total[rel] else 0
            rows.append(row)

    with open(f"{D}/eval/feature_analysis.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{'config':<13}{'feature':<15}{'pages':>9}{'%pages':>8}{'%any_rel':>9}"
          f"{'ret_manu':>9}{'ret_brand':>10}{'ret_made':>9}{'ret_comp':>9}")
    for r in rows:
        print(f"{r['config']:<13}{r['feature']:<15}{r['pages']:>9,}{r['share_of_pages']:>8.2f}"
              f"{r['pages_with_any_relation']:>9.2f}{r['retained_manufactures']:>9.2f}"
              f"{r['retained_has_brand']:>10.2f}{r['retained_made_in']:>9.2f}"
              f"{r['retained_is_component_of']:>9.2f}")
    print(f"\n[DONE] {D}/eval/feature_analysis.csv")


if __name__ == "__main__":
    main()
