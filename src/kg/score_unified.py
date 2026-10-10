"""
score_unified.py -- ONE scorer for every configuration (LLM and classical).

Run from the project root:
    python3 -m src.kg.score_unified        # reads data/interim/gold_annotation.xlsx
    python3 -m src.kg.score_unified --extra classical_svm=data/interim/preds_svm.jsonl

What it fixes compared with compare_all_four.py
  * all four LLM configurations are read from the RAW files (no normalized file)
  * one metric definition for every field and every system (positive-class P/R/F1)
  * matching is token-based, not character-substring ("in" no longer matches "india")
  * three matching modes are reported: strict, relaxed, binarized
  * page counts are reported separately from TP/FP/FN/TN
  * coverage is reported (gold pages, scored pages, missing predictions) and missing
    predictions are handled by an explicit policy instead of being skipped silently

Counting rule per page and field (value fields):
  gold has value, prediction has value, match      -> TP
  gold has value, prediction has value, no match   -> FP + FN  (wrong value)
  gold unknown,   prediction has value             -> FP
  gold has value, prediction unknown               -> FN
  both unknown                                     -> TN
manufactures: positive class is "yes" (yes vs. not-yes).

Missing-prediction policies (both reported):
  as_unknown  -- a page without a prediction counts as "unknown" for every field (main)
  scored_only -- pages without a prediction are left out

Gold sets:
  All gold labels come from ONE file, data/interim/gold_annotation.xlsx (437 rows).
  gold117 -- the 117 topup_* rows (relations only; company/product empty -> not scored)
  strat   -- the 320 strat_* rows (all six fields), reported unweighted and weighted to
             the corpus (1/inclusion probability, per field using the pages drawn for it)
  strat_visible -- strat with the BEYOND_3000 labels set to unknown, i.e. only
             information the models could see in the first 3,000 characters
  pooled  -- gold117 + strat, unweighted (relation fields only)

Outputs (data/interim/eval/):
  scores_long.csv     every system x gold set x policy x mode x field
  page_details.csv    per page, field and system: gold, prediction, outcome (for error analysis)
  coverage.csv        gold / scored / missing pages per system and gold set
"""

import argparse
import csv
import json
import os
import random
import re

from src.normalization.location_aliases import LOCATION_ALIASES

D = "data/interim"
CONFIGS = {
    "qwen_single": f"{D}/extraction_results_full_qwen.jsonl",
    "qwen_iter": f"{D}/extraction_results_iterative_full.jsonl",
    "llama_single": f"{D}/extraction_results_llama70b_full.jsonl",
    "llama_iter": f"{D}/extraction_results_iterative_llama_full.jsonl",
}
GOLD_FILE = f"{D}/gold_annotation.xlsx"
STRAT_REF = f"{D}/annotation_strat_reference_HIDDEN.jsonl"
OUT_DIR = f"{D}/eval"

REL_FIELDS = ["manufactures", "has_brand", "made_in", "is_component_of"]
ENT_FIELDS = ["company", "product"]
PRED_KEY = {"company": "company", "product": "product", "manufactures": "manufactures",
            "has_brand": "distributed_by", "made_in": "made_in",
            "is_component_of": "is_component_of"}
NEG = {"", "unknown", "none", "null", "n/a", "na", "not stated", "not specified"}
STOP = {"the", "and", "of", "for", "a", "an", "inc", "ltd", "co", "llc", "gmbh", "com"}
MODES = ["strict", "relaxed", "binarized"]


# ---------------------------------------------------------------- normalization
def clean(v):
    v = "" if v is None else str(v)
    v = v.strip().lower()
    v = re.sub(r"\s+", " ", v)
    return v.strip(" .,;:'\"")


def is_pos(field, v):
    v = clean(v)
    if field == "manufactures":
        return v.startswith("yes")
    return v not in NEG


def loc(v):
    return LOCATION_ALIASES.get(v, v).lower() if v else v


def alternatives(field, v):
    """All accepted surface forms of a gold or predicted value."""
    v = clean(v)
    alts = {v}
    for part in re.split(r"\s+/\s+", v):
        alts.add(part.strip())
    for inner in re.findall(r"\(([^)]*)\)", v):
        alts.add(inner.strip())
    alts.add(re.sub(r"\([^)]*\)", "", v).strip())
    if field == "made_in":
        alts = {loc(a) for a in alts}
    if field == "company":
        alts |= {re.sub(r"\.(com|net|org|de|co\.uk|com\.au|[a-z]{2})$", "", a) for a in list(alts)}
        alts |= {re.sub(r"[^a-z0-9]", "", a) for a in list(alts)}
    return {a for a in alts if a}


def tokens(v):
    return {t for t in re.findall(r"[a-z0-9]+", v) if t not in STOP}


def match(field, gold, pred, mode):
    if field == "manufactures" or mode == "binarized":
        return True  # only called when both are positive
    ga, pa = alternatives(field, gold), alternatives(field, pred)
    if ga & pa:
        return True
    if mode == "strict":
        return False
    for g in ga:
        for p in pa:
            gt, pt = tokens(g), tokens(p)
            small = gt if len(gt) <= len(pt) else pt
            if small and any(len(t) >= 3 for t in small) and (gt <= pt or pt <= gt):
                return True
    return False


def outcome(field, gold, pred, mode):
    gp, pp = is_pos(field, gold), is_pos(field, pred)
    if gp and pp:
        return "TP" if match(field, gold, pred, mode) else "WRONG"
    if pp:
        return "FP"
    if gp:
        return "FN"
    return "TN"


# ---------------------------------------------------------------- loading
def load_preds(path):
    preds, dup, bad = {}, 0, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            dom = str(r.get("domain", "")).strip()
            if not dom:
                continue
            if dom in preds:
                dup += 1
                continue
            preds[dom] = {fld: r.get(key, "unknown") for fld, key in PRED_KEY.items()}
    return preds, dup, bad


# Labels whose only evidence lies after character 3,000 (the models never saw it),
# found during annotation of the stratified sample. Used for the "strat_visible" view.
BEYOND_3000 = {
    "strat_0023": {"product", "is_component_of"}, "strat_0066": {"has_brand"},
    "strat_0070": {"product", "has_brand"}, "strat_0162": {"has_brand"},
    "strat_0166": {"has_brand"}, "strat_0205": {"made_in"},
    "strat_0213": {"manufactures", "made_in"}, "strat_0222": {"product", "has_brand"},
    "strat_0227": {"manufactures", "made_in"},
}


def load_gold(path):
    """One file, one sheet: record_id, domain, url, company, product, manufactures,
    has_brand, made_in, is_component_of. topup_* rows = the 117 (no company/product);
    strat_* rows = the 320 stratified pages (weights from the hidden reference file)."""
    from openpyxl import load_workbook
    ref = {}
    if os.path.exists(STRAT_REF):
        with open(STRAT_REF, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                ref[r["record_id"]] = r
    rows = list(load_workbook(path, read_only=True).worksheets[0].values)
    head = [str(h).strip() if h else "" for h in rows[0]]
    ix = {c: head.index(c) for c in head if c}
    gold117, strat = [], []
    for row in rows[1:]:
        rid = row[ix["record_id"]]
        if not rid:
            continue
        rid = str(rid).strip()
        gold = {}
        for f in ENT_FIELDS + REL_FIELDS:
            v = row[ix[f]]
            if v is not None and str(v).strip() != "":
                gold[f] = v
        page = {"record_id": rid, "domain": str(row[ix["domain"]]).strip(), "gold": gold,
                "beyond": BEYOND_3000.get(rid, set()), "weight": {}, "drawn_for": None}
        if rid.startswith("strat_"):
            rf = ref.get(rid)
            if rf:
                page["drawn_for"] = rf["drawn_for_field"]
                page["weight"] = {rf["drawn_for_field"]: 1.0 / rf["inclusion_prob"]}
            strat.append(page)
        else:
            gold117.append(page)
    return gold117, strat


# ---------------------------------------------------------------- scoring
def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def tally(items):
    """items: list of (outcome, weight). WRONG counts as one FP and one FN."""
    c = {"TP": 0.0, "FP": 0.0, "FN": 0.0, "TN": 0.0}
    for o, w in items:
        if o == "WRONG":
            c["FP"] += w
            c["FN"] += w
        else:
            c[o] += w
    return c


def bootstrap_f1(items_by_stratum, n_boot, seed):
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        sample = []
        for items in items_by_stratum.values():
            sample += [items[rng.randrange(len(items))] for _ in items]
        c = tally(sample)
        vals.append(prf(c["TP"], c["FP"], c["FN"])[2])
    vals.sort()
    return vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=GOLD_FILE, help="gold_annotation.xlsx")
    ap.add_argument("--extra", action="append", default=[],
                    help="name=path.jsonl with {domain, company, product, manufactures, "
                         "distributed_by, made_in, is_component_of}")
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()

    systems = dict(CONFIGS)
    for e in args.extra:
        name, path = e.split("=", 1)
        systems[name] = path

    preds = {}
    for name, path in systems.items():
        preds[name], dup, bad = load_preds(path)
        print(f"[LOAD] {name}: {len(preds[name]):,} pages ({dup} duplicate domains, "
              f"{bad} unparseable lines)")

    g117, strat = load_gold(args.gold)
    print(f"[LOAD] gold: {len(g117)} top-up pages, {len(strat)} stratified pages "
          f"({sum(1 for p in strat if p['weight'])} with sampling weights)")
    gold_sets = {"gold117": (g117, REL_FIELDS), "strat": (strat, ENT_FIELDS + REL_FIELDS)}
    vis = []
    for p in strat:
        q = dict(p, gold=dict(p["gold"]))
        for f in p["beyond"]:
            if f in q["gold"]:
                q["gold"][f] = "unknown"
        vis.append(q)
    gold_sets["strat_visible"] = (vis, ENT_FIELDS + REL_FIELDS)
    gold_sets["pooled"] = (g117 + strat, REL_FIELDS)

    os.makedirs(OUT_DIR, exist_ok=True)
    long_rows, detail_rows, cov_rows = [], [], []

    for gname, (pages, fields) in gold_sets.items():
        for sname, P in preds.items():
            missing = [p for p in pages if p["domain"] not in P]
            cov_rows.append({"gold_set": gname, "system": sname, "gold_pages": len(pages),
                             "scored_pages": len(pages) - len(missing),
                             "missing_pages": len(missing)})
            for policy in ("as_unknown", "scored_only"):
                use = pages if policy == "as_unknown" else [p for p in pages if p["domain"] in P]
                for mode in MODES:
                    for f in fields:
                        items, strata = [], {}
                        for p in use:
                            if f not in p["gold"]:
                                continue
                            pred = P.get(p["domain"], {}).get(f, "unknown")
                            o = outcome(f, p["gold"][f], pred, mode)
                            items.append((o, 1.0))
                            strata.setdefault("all", []).append((o, 1.0))
                            if policy == "as_unknown" and mode == "relaxed":
                                detail_rows.append({"gold_set": gname, "system": sname,
                                                    "record_id": p["record_id"],
                                                    "domain": p["domain"], "field": f,
                                                    "gold": p["gold"][f], "pred": pred,
                                                    "outcome": o,
                                                    "missing_pred": p["domain"] not in P})
                        c = tally(items)
                        pr, rc, f1 = prf(c["TP"], c["FP"], c["FN"])
                        lo, hi = bootstrap_f1(strata, args.boot, 42) if items else (0, 0)
                        row = {"gold_set": gname, "system": sname, "policy": policy,
                               "mode": mode, "field": f, "weighting": "unweighted",
                               "n_pages": len(items),
                               "gold_positive": sum(is_pos(f, p["gold"][f]) for p in use if f in p["gold"]),
                               **{k: round(v, 3) for k, v in c.items()},
                               "precision": round(pr, 3), "recall": round(rc, 3),
                               "f1": round(f1, 3), "f1_ci_low": round(lo, 3),
                               "f1_ci_high": round(hi, 3)}
                        long_rows.append(row)
                        # corpus-weighted estimate (strat sets, pages drawn for this field)
                        if gname in ("strat", "strat_visible"):
                            wi, ws = [], {}
                            for p in use:
                                if f not in p["weight"] or f not in p["gold"]:
                                    continue
                                pred = P.get(p["domain"], {}).get(f, "unknown")
                                o = outcome(f, p["gold"][f], pred, mode)
                                w = p["weight"][f]
                                wi.append((o, w))
                                ws.setdefault(w, []).append((o, w))
                            if wi:
                                cw = tally(wi)
                                pr, rc, f1 = prf(cw["TP"], cw["FP"], cw["FN"])
                                lo, hi = bootstrap_f1(ws, args.boot, 42)
                                long_rows.append({**row, "weighting": "corpus_weighted",
                                                  "n_pages": len(wi),
                                                  **{k: round(v, 1) for k, v in cw.items()},
                                                  "precision": round(pr, 3),
                                                  "recall": round(rc, 3), "f1": round(f1, 3),
                                                  "f1_ci_low": round(lo, 3),
                                                  "f1_ci_high": round(hi, 3)})

    def write(path, rows):
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    write(f"{OUT_DIR}/scores_long.csv", long_rows)
    write(f"{OUT_DIR}/page_details.csv", detail_rows)
    write(f"{OUT_DIR}/coverage.csv", cov_rows)

    # console summary: main setting
    print("\nCOVERAGE")
    for r in cov_rows:
        print(f"  {r['gold_set']:<14}{r['system']:<16}gold={r['gold_pages']:<5}"
              f"scored={r['scored_pages']:<5}missing={r['missing_pages']}")
    for gname in gold_sets:
        for weighting in ("unweighted", "corpus_weighted"):
            sel = [r for r in long_rows if r["gold_set"] == gname and r["policy"] == "as_unknown"
                   and r["weighting"] == weighting]
            if not sel:
                continue
            print(f"\nF1  [{gname} | missing=as_unknown | {weighting}]   strict / relaxed / binarized")
            for f in gold_sets[gname][1]:
                line = f"  {f:<16}"
                for sname in preds:
                    v = {r["mode"]: r["f1"] for r in sel if r["field"] == f and r["system"] == sname}
                    if not v:
                        line += f"{sname}: n/a   "
                        continue
                    line += f"{sname}: {v['strict']:.3f}/{v['relaxed']:.3f}/{v['binarized']:.3f}   "
                print(line)
    print(f"\n[DONE] {OUT_DIR}/scores_long.csv, page_details.csv, coverage.csv")


if __name__ == "__main__":
    main()
