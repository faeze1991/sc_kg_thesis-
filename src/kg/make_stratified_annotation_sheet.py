"""
make_stratified_annotation_sheet.py

Builds a STRATIFIED annotation sample so every relation field gets enough
positives AND enough pages where the models said "unknown" (needed for recall).

For each field f in (manufactures, has_brand, made_in, is_component_of):
  POS_f = pages where at least one of the 4 configurations predicted a value
  NEG_f = pages where none of the 4 configurations predicted a value
          (a missing/errored prediction counts as "no value")
  -> draw N_POS pages from POS_f and N_NEG pages from NEG_f.

Each page is drawn once. Stratum sizes and inclusion probabilities go to the
HIDDEN reference file, so corpus-level (weighted) precision/recall can be
computed later. The Excel sheet shows no model output, no field and no stratum,
and rows are shuffled, so the annotation stays blind.

Excluded: all domains of the earlier annotation rounds (data/annotation/silver/).

Usage (from project root, venv active):
    python -m src.kg.make_stratified_annotation_sheet            # 40 pos + 40 neg per field (seed 2026)
    python -m src.kg.make_stratified_annotation_sheet 30 30      # smaller
    python -m src.kg.make_stratified_annotation_sheet 40 40 7    # other seed

Outputs:
    data/annotation/annotation_strat_BLIND.xlsx          <- annotate this
    data/annotation/annotation_strat_reference_HIDDEN.jsonl  <- do NOT open before annotating
    data/annotation/annotation_strat_design.json         <- stratum sizes (safe to open)
    data/annotation/annotation_strat_fulltext/<id>.txt   <- only for pages over Excel's cell limit
"""

import csv
import json
import os
import random
import sys

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

N_POS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
N_NEG = int(sys.argv[2]) if len(sys.argv) > 2 else 40
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 2026

D = "data/interim"
PAGES = f"{D}/fetched_pages.jsonl"
CONFIGS = {  # raw (un-normalized) outputs of all four configurations
    "qwen_single": f"{D}/extraction_results_full_qwen.jsonl",
    "qwen_iter": f"{D}/extraction_results_iterative_full.jsonl",
    "llama_single": f"{D}/extraction_results_llama70b_full.jsonl",
    "llama_iter": f"{D}/extraction_results_iterative_llama_full.jsonl",
}
A = "data/annotation"
# the 1,166 pages of the earlier (silver) annotation rounds are excluded from the gold sample
USED = [f"{A}/silver/annotation_sample_WITH_CLAUDE.csv", f"{A}/silver/annotation_topup_WITH_CLAUDE.csv"]

OUT_XLSX = f"{A}/annotation_strat_BLIND.xlsx"
OUT_REF = f"{A}/annotation_strat_reference_HIDDEN.jsonl"
OUT_DESIGN = f"{A}/annotation_strat_design.json"
OUT_FULLTEXT_DIR = f"{A}/annotation_strat_fulltext"

# field name used in the thesis -> key in the raw extraction files
FIELDS = {
    "manufactures": "manufactures",
    "has_brand": "distributed_by",
    "made_in": "made_in",
    "is_component_of": "is_component_of",
}
EXCEL_CELL_LIMIT = 32767
TEXT_KEEP = 32000
NEGATIVE_VALUES = {"", "unknown", "none", "null", "n/a"}


def is_positive(field, value):
    v = str(value if value is not None else "").strip().lower()
    if field == "manufactures":
        return v == "yes"
    return v not in NEGATIVE_VALUES


def load_used_domains():
    used = set()
    for path in USED:
        with open(path, newline="", encoding="utf-8") as f:
            sample = f.read(4096)
            f.seek(0)
            delim = ";" if sample.count(";") > sample.count(",") else ","
            used.update((r.get("domain") or "").strip() for r in csv.DictReader(f, delimiter=delim))
    used = {d for d in used if "." in d}  # drop empty / malformed cells of the semicolon CSVs
    return used


def main():
    random.seed(SEED)
    used = load_used_domains()
    print(f"[INFO] {len(used):,} previously annotated domains excluded")

    # 1. universe = all fetched pages with text, minus used domains
    universe = set()
    with open(PAGES, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") == "found" and r.get("text") and r["domain"] not in used:
                universe.add(r["domain"])
    print(f"[INFO] {len(universe):,} eligible pages")

    # 2. predictions of all four configurations (raw values kept for the hidden file)
    preds = {name: {} for name in CONFIGS}
    for name, path in CONFIGS.items():
        n_bad = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    n_bad += 1
                    continue
                dom = r.get("domain", "").strip()
                if dom in universe:
                    preds[name][dom] = {fld: r.get(key, "unknown") for fld, key in FIELDS.items()}
        print(f"[INFO] {name}: {len(preds[name]):,} predictions on eligible pages "
              f"({len(universe) - len(preds[name]):,} missing, {n_bad} unparseable lines)")

    # 3. strata per field
    pos = {fld: set() for fld in FIELDS}
    for name in CONFIGS:
        for dom, p in preds[name].items():
            for fld in FIELDS:
                if is_positive(fld, p[fld]):
                    pos[fld].add(dom)

    # 4. draw, field by field, without reusing a page
    taken = set()
    draws = []  # (domain, field, stratum)
    design = {"seed": SEED, "n_pos": N_POS, "n_neg": N_NEG,
              "universe_size": len(universe), "strata": {}}
    for fld in FIELDS:
        pos_pool = sorted(pos[fld] - taken)
        neg_pool = sorted((universe - pos[fld]) - taken)
        k_pos, k_neg = min(N_POS, len(pos_pool)), min(N_NEG, len(neg_pool))
        if k_pos < N_POS:
            print(f"[WARN] {fld}: only {k_pos} positive pages available")
        got_pos = random.sample(pos_pool, k_pos)
        got_neg = random.sample(neg_pool, k_neg)
        taken.update(got_pos)
        taken.update(got_neg)
        draws += [(d, fld, "pos") for d in got_pos] + [(d, fld, "neg") for d in got_neg]
        design["strata"][fld] = {
            "pos_size": len(pos[fld]), "neg_size": len(universe) - len(pos[fld]),
            "pos_drawn": k_pos, "neg_drawn": k_neg,
        }
        print(f"[INFO] {fld}: POS={len(pos[fld]):,} NEG={len(universe) - len(pos[fld]):,} "
              f"-> drew {k_pos} + {k_neg}")

    random.shuffle(draws)  # row order reveals nothing

    # 5. page text for the drawn pages only
    wanted = {d for d, _, _ in draws}
    page = {}
    with open(PAGES, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("domain") in wanted and r["domain"] not in page:
                page[r["domain"]] = {"url": r.get("url", ""), "text": r.get("text", "")}

    # 6. hidden reference file
    with open(OUT_REF, "w", encoding="utf-8") as f:
        for i, (dom, fld, stratum) in enumerate(draws):
            s = design["strata"][fld]
            size = s["pos_size"] if stratum == "pos" else s["neg_size"]
            drawn = s["pos_drawn"] if stratum == "pos" else s["neg_drawn"]
            f.write(json.dumps({
                "record_id": f"strat_{i:04d}", "domain": dom,
                "drawn_for_field": fld, "stratum": stratum,
                "stratum_size": size, "inclusion_prob": drawn / size,
                "in_pos": {x: dom in pos[x] for x in FIELDS},
                "predictions": {name: preds[name].get(dom) for name in CONFIGS},
            }, ensure_ascii=False) + "\n")
    with open(OUT_DESIGN, "w", encoding="utf-8") as f:
        json.dump(design, f, indent=2)

    # 7. Excel workbook
    wb = Workbook()
    ins = wb.active
    ins.title = "Instructions"
    arial, bold = Font(name="Arial", size=10), Font(name="Arial", size=10, bold=True)
    lines = [
        ("HOW TO ANNOTATE", ""),
        ("Edit only", "The yellow columns on the 'Annotation' sheet (gold_* and annotator_notes)."),
        ("Source", "Read the page_text (full page, same as for the 117 gold records). Label only what the text states explicitly."),
        ("unknown", "If the text does not state it, write exactly: unknown"),
        ("", ""),
        ("FIELD DEFINITIONS (same as the extraction prompt)", ""),
        ("gold_manufactures", "yes / no / unknown: does the text indicate that the company operating the site manufactures the product itself?"),
        ("gold_has_brand", "If the product is made by a DIFFERENT company or brand than the site operator, name that company or brand. Otherwise unknown. (Old name: distributed_by.)"),
        ("gold_made_in", "Country or location where the product was made, if stated. Otherwise unknown."),
        ("gold_is_component_of", "Only if the product is explicitly described as a component/part of a larger product: name the larger product (short name). Otherwise unknown."),
        ("annotator_notes", "Optional: doubts, e.g. 'yes (debatable)'. Keep the gold cell itself clean."),
        ("", ""),
        ("EXAMPLE ROW (format only, not part of the sample)", ""),
        ("gold_manufactures", "no"),
        ("gold_has_brand", "festool"),
        ("gold_made_in", "germany"),
        ("gold_is_component_of", "festool ks 120 eb"),
        ("", ""),
        ("Long pages", "If page_text ends with [TRUNCATED], the full text is in data/annotation/annotation_strat_fulltext/<record_id>.txt"),
    ]
    for a, b in lines:
        ins.append([a, b])
    for row in ins.iter_rows():
        for c in row:
            c.font = arial
            c.alignment = Alignment(wrap_text=True, vertical="top")
    for r in (1, 6, 13):
        ins.cell(row=r, column=1).font = bold
    ins.column_dimensions["A"].width = 32
    ins.column_dimensions["B"].width = 110

    ws = wb.create_sheet("Annotation")
    cols = ["record_id", "domain", "url", "gold_manufactures", "gold_has_brand",
            "gold_made_in", "gold_is_component_of", "annotator_notes", "page_text"]
    ws.append(cols)
    os.makedirs(OUT_FULLTEXT_DIR, exist_ok=True)
    n_trunc = 0
    for i, (dom, _, _) in enumerate(draws):
        rid = f"strat_{i:04d}"
        p = page.get(dom, {"url": "", "text": ""})
        text = p["text"]
        if len(text) > EXCEL_CELL_LIMIT:
            with open(f"{OUT_FULLTEXT_DIR}/{rid}.txt", "w", encoding="utf-8") as f:
                f.write(text)
            text = text[:TEXT_KEEP] + " [TRUNCATED]"
            n_trunc += 1
        ws.append([rid, dom, p["url"], "", "", "", "", "", text])

    yellow = PatternFill("solid", start_color="FFF2CC")
    widths = [12, 28, 40, 16, 22, 18, 26, 24, 100]
    for j, w in enumerate(widths, start=1):
        ws.column_dimensions[ws.cell(row=1, column=j).column_letter].width = w
    for row in ws.iter_rows():
        for c in row:
            c.font = bold if c.row == 1 else arial
            c.alignment = Alignment(wrap_text=(c.column == 9), vertical="top")
            if 4 <= c.column <= 8 and c.row > 1:
                c.fill = yellow
    ws.freeze_panes = "D2"
    dv = DataValidation(type="list", formula1='"yes,no,unknown"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"D2:D{len(draws) + 1}")
    wb.save(OUT_XLSX)

    print(f"[DONE] {len(draws)} pages -> {OUT_XLSX}")
    print(f"[DONE] hidden reference -> {OUT_REF} (do not open before annotating)")
    print(f"[DONE] design -> {OUT_DESIGN}; {n_trunc} pages over the cell limit -> {OUT_FULLTEXT_DIR}/")


if __name__ == "__main__":
    main()
