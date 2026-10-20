"""
collect_thesis_numbers.py -- gathers every number and chart needed for Sections
3.7-3.10, Chapter 4 and Chapter 5 into ONE text file plus PNG figures.
Read-only: it does not change any data. No LLM calls.

Run from the project root (venv active):
    python -m src.kg.collect_thesis_numbers > data/interim/eval/thesis_numbers.txt 2>&1

Each section is independent: if one fails, it prints the error and continues.
"""

import csv
import json
import os
import random
import traceback
from collections import Counter, defaultdict

D = "data/interim"
EV = f"{D}/eval"
KG = f"{D}/kg"
FIG = f"{EV}/figures"
CONFIGS = ["qwen_single", "qwen_iter", "llama_single", "llama_iter"]
LABEL = {"qwen_single": "Qwen single", "qwen_iter": "Qwen iter", "llama_single": "Llama single",
         "llama_iter": "Llama iter", "classical_svm": "SVM", "classical_lr": "LogReg"}
RAW = {"qwen_single": f"{D}/extraction_results_full_qwen.jsonl",
       "qwen_iter": f"{D}/extraction_results_iterative_full.jsonl",
       "llama_single": f"{D}/extraction_results_llama70b_full.jsonl",
       "llama_iter": f"{D}/extraction_results_iterative_llama_full.jsonl"}
NORM = {"qwen_single": f"{D}/extraction_results_normalized_FINAL.jsonl",
        "qwen_iter": f"{D}/extraction_results_iterative_normalized.jsonl",
        "llama_single": f"{D}/extraction_results_llama70b_normalized.jsonl",
        "llama_iter": f"{D}/extraction_results_iterative_llama_normalized.jsonl"}
FIELDS = ["company", "product", "manufactures", "has_brand", "made_in", "is_component_of"]
NEG = {"", "unknown", "none", "null", "n/a", "na"}


def section(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def run(fn):
    try:
        fn()
    except Exception:
        print(f"[ERROR in {fn.__name__}]")
        traceback.print_exc()


def read_csv(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def nlines(p):
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8", errors="replace") as f:
        return sum(1 for _ in f)


# ----------------------------------------------------------------------------- 3.7
def s37_runs():
    section("3.7  RUN OUTPUTS (records per configuration, benchmark/subset files)")
    for c, p in RAW.items():
        print(f"  {LABEL[c]:<13} {nlines(p):>9,} records   ({p})")
    for p in [f"{D}/extraction_results_bench2k.jsonl", f"{D}/extraction_results_llama70b_bench2k.jsonl",
              f"{D}/extraction_results_gptoss20b_bench2k.jsonl", f"{D}/extraction_results_mistral_300.jsonl"]:
        n = nlines(p)
        print(f"  subset  {os.path.basename(p):<45} {n if n is not None else 'missing'}")
    p = f"{EV}/audit_summary.json"
    if os.path.exists(p):
        s = json.load(open(p))
        print(f"\n  fetched pages with text: {s.get('universe'):,}")
        for c, v in s.get("configs", {}).items():
            print(f"  {LABEL.get(c, c):<13} missing {v['missing_pages']:>6,} | unparseable {v['unparseable_lines']} | "
                  f"records with schema problems {v['records_with_any_problem']:,} | suspicious {v['suspicious_values']}")
        for c, v in s.get("logs", {}).items():
            print(f"  LOG {LABEL.get(c, c):<13} {json.dumps({k: v[k] for k in v if k != 'path'})}")
    else:
        print("  (audit_summary.json not found -- run: python3 -m src.kg.audit_outputs)")


# ----------------------------------------------------------------------------- 3.8
def s38_normalization():
    section("3.8  NORMALIZATION EFFECT (distinct values before -> after, non-unknown only)")
    for c, p in NORM.items():
        if not os.path.exists(p):
            print(f"  {LABEL[c]}: normalized file missing")
            continue
        cnt = defaultdict(set)
        n = 0
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                n += 1
                for raw, out in (("company_cleaned", "company_canonical"),
                                 ("distributed_by_cleaned", "distributed_by_canonical"),
                                 ("made_in_raw", "made_in")):
                    a, b = str(r.get(raw) or ""), str(r.get(out) or "")
                    if a.strip().lower() not in NEG:
                        cnt[raw].add(a.strip().lower())
                    if b.strip().lower() not in NEG:
                        cnt[out].add(b.strip().lower())
        print(f"  {LABEL[c]:<13} records {n:,} | company {len(cnt['company_cleaned']):,} -> "
              f"{len(cnt['company_canonical']):,} | brand {len(cnt['distributed_by_cleaned']):,} -> "
              f"{len(cnt['distributed_by_canonical']):,} | made_in {len(cnt['made_in_raw']):,} -> "
              f"{len(cnt['made_in']):,}")


# ----------------------------------------------------------------------------- 3.9
def s39_graphs():
    section("3.9  KNOWLEDGE GRAPHS (final version, from data/interim/kg/summary.csv)")
    rows = read_csv(f"{KG}/summary.csv")
    keys = ["pages", "nodes", "edges", "nodes_company", "nodes_product", "nodes_location",
            "products_source", "products_canonical", "products_parent", "products_parent_family_or_unspecific",
            "edges_offers", "edges_has_brand", "edges_manufactures", "edges_made_in", "edges_is_component_of",
            "route1_clusters_accepted", "route1_pages_merged", "route1_clusters_rejected",
            "route1_rejected_brand_conflict", "route1_rejected_unspecific", "route1_rejected_member_conflict",
            "route1_members_withheld_unspecific", "route2_groups_accepted", "route2_pages_merged",
            "route2_names_with_different_brands", "route3_candidates", "page_brand_conflicts",
            "made_in_country", "made_in_subdivision", "made_in_region", "made_in_unresolved",
            "parent_linked_to_product", "parent_shared_specific", "parent_page_scoped", "parent_generic_skipped",
            "pages_without_product", "cq5_multi_origin_products", "cq6_chains", "cq7_chains",
            "cq8_candidate_pairs"]
    for set_name in ("full", "common"):
        sel = {r["config"]: r for r in rows if r["page_set"] == set_name}
        print(f"\n  [{set_name} page set]")
        print(f"  {'':<40}" + "".join(f"{LABEL[c]:>14}" for c in CONFIGS))
        for k in keys:
            print(f"  {k:<40}" + "".join(f"{sel.get(c, {}).get(k, ''):>14}" for c in CONFIGS))
        top = {c: sel.get(c, {}).get("unresolved_top", "") for c in CONFIGS}
        for c in CONFIGS:
            print(f"  unresolved made_in top values ({LABEL[c]}): {top[c][:300]}")
    for v in ("summary_v1.csv", "summary_v2.csv"):
        p = f"{KG}/{v}"
        if os.path.exists(p):
            r = {x["config"]: x for x in read_csv(p) if x["page_set"] == "full"}
            print(f"\n  CQ8 candidates in {v}: " + ", ".join(f"{LABEL[c]} {r.get(c, {}).get('cq8_candidate_pairs')}"
                                                         for c in CONFIGS))


def s39_cq_examples():
    section("3.9 / APPENDIX B  CQ ANSWERS (Python) AND SPARQL CHECK")
    for c in ("qwen_single", "llama_single"):
        p = f"{KG}/cq_{c}_full.txt"
        if os.path.exists(p):
            print(f"\n----- {p}")
            print(open(p, encoding="utf-8").read()[:5000])
    for c in CONFIGS:
        p = f"{KG}/sparql_{c}_full.txt"
        if os.path.exists(p):
            txt = open(p, encoding="utf-8").read()
            print(f"\n----- {p} (first line + check)")
            print(txt.splitlines()[0])
            print("\n".join(l for l in txt.splitlines() if "SPARQL" in l and "Python" in l))


# ----------------------------------------------------------------------------- 4 / 5
def s5_scores():
    section("5  EXTRACTION SCORES (gold = 320 stratified pages; missing = as_unknown)")
    rows = read_csv(f"{EV}/scores_long.csv")
    systems = []
    for r in rows:
        if r["system"] not in systems:
            systems.append(r["system"])
    for gset, weighting in (("strat", "unweighted"), ("strat", "corpus_weighted"), ("strat_visible", "unweighted")):
        for mode in ("strict", "relaxed", "binarized"):
            sel = [r for r in rows if r["gold_set"] == gset and r["weighting"] == weighting
                   and r["policy"] == "as_unknown" and r["mode"] == mode]
            if not sel:
                continue
            print(f"\n  [{gset} | {weighting} | {mode}]  F1 [95% CI]  (P / R)")
            for f in FIELDS:
                line = f"  {f:<16}"
                for s in systems:
                    x = [r for r in sel if r["system"] == s and r["field"] == f]
                    if x:
                        x = x[0]
                        line += f" {LABEL.get(s, s)}: {float(x['f1']):.2f} [{float(x['f1_ci_low']):.2f}-" \
                                f"{float(x['f1_ci_high']):.2f}] ({float(x['precision']):.2f}/{float(x['recall']):.2f}) |"
                print(line)
    gp = {}
    for r in rows:
        if r["gold_set"] == "strat" and r["weighting"] == "unweighted" and r["mode"] == "relaxed" \
                and r["policy"] == "as_unknown":
            gp[r["field"]] = r["gold_positive"]
    print(f"\n  gold positives per field (320 pages): {gp}")
    sel = [r for r in rows if r["gold_set"] == "strat" and r["weighting"] == "unweighted"
           and r["mode"] == "relaxed" and r["policy"] == "scored_only"]
    print("\n  [scored_only policy, relaxed F1] " + " | ".join(
        f"{LABEL.get(r['system'], r['system'])}/{r['field']}: {float(r['f1']):.2f}" for r in sel
        if r["field"] in ("has_brand", "made_in")))
    cov = read_csv(f"{EV}/coverage.csv")
    print("\n  coverage: " + " | ".join(f"{r['gold_set']}/{LABEL.get(r['system'], r['system'])}: "
                                       f"missing {r['missing_pages']}" for r in cov if r["gold_set"] == "strat"))


def s5_paired():
    section("5  PAIRED BOOTSTRAP: F1 difference between configurations (relaxed, 320 pages, 2000 resamples)")
    rows = [r for r in read_csv(f"{EV}/page_details.csv") if r["gold_set"] == "strat"]
    by = defaultdict(dict)
    for r in rows:
        by[(r["field"], r["record_id"])][r["system"]] = r["outcome"]

    def f1(outs):
        tp = sum(o == "TP" for o in outs)
        fp = sum(o in ("FP", "WRONG") for o in outs)
        fn = sum(o in ("FN", "WRONG") for o in outs)
        return 2 * tp / (2 * tp + fp + fn) if tp else 0.0

    pairs = [("qwen_single", "qwen_iter"), ("llama_single", "llama_iter"),
             ("qwen_single", "llama_single"), ("qwen_iter", "llama_iter")]
    rnd = random.Random(42)
    for f in FIELDS:
        ids = sorted({rid for (ff, rid) in by if ff == f})
        line = f"  {f:<16}"
        for a, b in pairs:
            data = [(by[(f, i)].get(a), by[(f, i)].get(b)) for i in ids
                    if a in by[(f, i)] and b in by[(f, i)]]
            if not data:
                continue
            obs = f1([x for x, _ in data]) - f1([y for _, y in data])
            diffs = []
            for _ in range(2000):
                s = [data[rnd.randrange(len(data))] for _ in data]
                diffs.append(f1([x for x, _ in s]) - f1([y for _, y in s]))
            diffs.sort()
            lo, hi = diffs[50], diffs[1949]
            sig = "*" if lo > 0 or hi < 0 else " "
            line += f" {LABEL[a]}-{LABEL[b]}: {obs:+.2f} [{lo:+.2f},{hi:+.2f}]{sig} |"
        print(line)
    print("  (* = 95% interval excludes 0)")


def s5_features():
    section("RQ1  FEATURE ANALYSIS (pre-filter yield)")
    for r in read_csv(f"{EV}/feature_analysis.csv"):
        print("  " + " | ".join(f"{k}={v}" for k, v in r.items()))


def s5_audit():
    section("5  GRAPH AUDIT PRECISION (from your checked audit files)")
    from openpyxl import load_workbook
    for path, sheets in ((f"{KG}/audit_sheets_checked.xlsx", None), (f"{KG}/audit_cq8_v2_checked.xlsx", None)):
        if not os.path.exists(path):
            print(f"  {path} not found")
            continue
        wb = load_workbook(path, read_only=True)
        for ws in wb.worksheets:
            rows = list(ws.values)
            if not rows:
                continue
            h = [str(x) for x in rows[0]]
            if "verdict" not in h:
                continue
            vi = h.index("verdict")
            ci = h.index("config") if "config" in h else None
            st = defaultdict(Counter)
            for r in rows[1:]:
                if not r or not r[0]:
                    continue
                st[r[ci] if ci is not None else "all"][str(r[vi]).strip().lower()] += 1
            print(f"\n  {os.path.basename(path)} / {ws.title}")
            tot = Counter()
            for c, x in st.items():
                tot += x
                p = x["correct"] / (x["correct"] + x["incorrect"]) if x["correct"] + x["incorrect"] else 0
                print(f"    {LABEL.get(c, c):<13} {dict(x)}  precision {p:.2f}")
            if len(st) > 1:
                p = tot["correct"] / (tot["correct"] + tot["incorrect"])
                print(f"    {'ALL':<13} {dict(tot)}  precision {p:.2f}")
    p = f"{KG}/cq8_recheck.csv"
    if os.path.exists(p):
        t = Counter((r["verdict"], r["retained"]) for r in read_csv(p))
        print(f"\n  CQ8 recheck (audited v1 pairs in final graph): {dict(t)}")


# ----------------------------------------------------------------------------- figures
def figures():
    section("FIGURES -> data/interim/eval/figures/")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG, exist_ok=True)
    rows = read_csv(f"{EV}/scores_long.csv")
    sel = [r for r in rows if r["gold_set"] == "strat" and r["weighting"] == "unweighted"
           and r["policy"] == "as_unknown"]

    # Fig 1: relaxed F1 with CI, 4 LLM configurations
    fig, ax = plt.subplots(figsize=(10, 4.5))
    w = 0.2
    for i, c in enumerate(CONFIGS):
        ys, lo, hi = [], [], []
        for f in FIELDS:
            x = [r for r in sel if r["system"] == c and r["field"] == f and r["mode"] == "relaxed"][0]
            ys.append(float(x["f1"]))
            lo.append(float(x["f1"]) - float(x["f1_ci_low"]))
            hi.append(float(x["f1_ci_high"]) - float(x["f1"]))
        ax.bar([j + (i - 1.5) * w for j in range(len(FIELDS))], ys, w, yerr=[lo, hi], capsize=2, label=LABEL[c])
    ax.set_xticks(range(len(FIELDS)))
    ax.set_xticklabels(FIELDS)
    ax.set_ylabel("F1 (relaxed)")
    ax.set_ylim(0, 1)
    ax.legend(ncol=4, fontsize=8)
    ax.set_title("Extraction F1 on the 320-page gold sample (95% bootstrap CI)")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig_f1_llm_configs.png", dpi=200)
    print("  fig_f1_llm_configs.png")

    # Fig 2: binarized relation F1, LLMs vs classical baseline
    rels = ["manufactures", "has_brand", "made_in", "is_component_of"]
    systems = CONFIGS + [s for s in ("classical_svm", "classical_lr") if any(r["system"] == s for r in sel)]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    w = 0.8 / len(systems)
    for i, s in enumerate(systems):
        ys = [float([r for r in sel if r["system"] == s and r["field"] == f and r["mode"] == "binarized"][0]["f1"])
              for f in rels]
        ax.bar([j + (i - (len(systems) - 1) / 2) * w for j in range(len(rels))], ys, w, label=LABEL.get(s, s))
    ax.set_xticks(range(len(rels)))
    ax.set_xticklabels(rels)
    ax.set_ylabel("F1 (binarized)")
    ax.set_ylim(0, 1)
    ax.legend(ncol=3, fontsize=8)
    ax.set_title("Relation detection: LLM configurations vs. classical baselines")
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig_llm_vs_baseline.png", dpi=200)
    print("  fig_llm_vs_baseline.png")

    # Fig 3: CQ8 candidates v1 -> v2 -> v3
    vals = {}
    for tag, p in (("original", f"{KG}/summary_v1.csv"), ("rule 1", f"{KG}/summary_v2.csv"),
                   ("final", f"{KG}/summary.csv")):
        if os.path.exists(p):
            r = {x["config"]: x for x in read_csv(p) if x["page_set"] == "full"}
            vals[tag] = [int(r.get(c, {}).get("cq8_candidate_pairs") or 0) for c in CONFIGS]
    if vals:
        fig, ax = plt.subplots(figsize=(8, 4))
        w = 0.8 / len(vals)
        for i, (tag, ys) in enumerate(vals.items()):
            ax.bar([j + (i - (len(vals) - 1) / 2) * w for j in range(len(CONFIGS))], ys, w, label=tag)
        ax.set_xticks(range(len(CONFIGS)))
        ax.set_xticklabels([LABEL[c] for c in CONFIGS])
        ax.set_ylabel("CQ8 candidate pairs")
        ax.legend(fontsize=8)
        ax.set_title("CQ8 candidate company pairs after each identity-rule tightening")
        fig.tight_layout()
        fig.savefig(f"{FIG}/fig_cq8_versions.png", dpi=200)
        print("  fig_cq8_versions.png")

    # Fig 4: feature analysis (Qwen single)
    fa = [r for r in read_csv(f"{EV}/feature_analysis.csv") if r["config"] == "qwen_single"]
    if fa:
        fig, ax = plt.subplots(figsize=(9, 4))
        feats = [r["feature"] for r in fa]
        ax.bar(range(len(fa)), [float(r["share_of_pages"]) for r in fa], 0.4, label="share of pages kept")
        ax.bar([j + 0.4 for j in range(len(fa))], [float(r["pages_with_any_relation"]) for r in fa], 0.4,
               label="pages with any predicted relation")
        ax.set_xticks([j + 0.2 for j in range(len(fa))])
        ax.set_xticklabels(feats, rotation=20, fontsize=8)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.set_title("Pre-filter features: pages kept vs. relation yield (Qwen single-pass)")
        fig.tight_layout()
        fig.savefig(f"{FIG}/fig_feature_analysis.png", dpi=200)
        print("  fig_feature_analysis.png")


if __name__ == "__main__":
    for fn in (s37_runs, s38_normalization, s39_graphs, s39_cq_examples, s5_scores, s5_paired,
               s5_features, s5_audit, figures):
        run(fn)
    print("\n[DONE]")
