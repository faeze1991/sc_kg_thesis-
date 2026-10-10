"""
audit_outputs.py -- schema validation and failure audit of the SAVED extraction outputs.
No model calls. Run from the project root:
    python3 -m src.kg.audit_outputs

For each of the four raw output files it reports:
  * lines, unparseable lines, duplicate domains, status values
  * records missing a required field, non-string values, empty strings
  * manufactures values outside {yes, no, unknown}
  * suspicious values: longer than 80 characters or sentence-like (likely explanations
    instead of short answers)
  * coverage against all fetched pages (status=found with text): which pages are missing
And it joins the available failure logs:
  * extraction_skipped_iterative_full.log (Qwen iterative, JSON lines with error text)
  * extraction_skipped_llama70b.log (tab format; its date matches the Llama 2k benchmark,
    so overlap with the full-run missing pages is reported, not assumed)

Outputs (data/interim/eval/):
  audit_summary.json
  audit_missing_<config>.txt        domains without any record
  audit_invalid_<config>.csv        every record with a schema problem or suspicious value
"""

import csv
import json
import os
import re
from collections import Counter

D = "data/interim"
OUT = f"{D}/eval"
PAGES = f"{D}/fetched_pages.jsonl"
CONFIGS = {
    "qwen_single": f"{D}/extraction_results_full_qwen.jsonl",
    "qwen_iter": f"{D}/extraction_results_iterative_full.jsonl",
    "llama_single": f"{D}/extraction_results_llama70b_full.jsonl",
    "llama_iter": f"{D}/extraction_results_iterative_llama_full.jsonl",
}
LOGS = {
    "qwen_iter": f"{D}/extraction_skipped_iterative_full.log",
    "llama_single(?)": f"{D}/extraction_skipped_llama70b.log",
}
REQUIRED = ["company", "product", "manufactures", "distributed_by", "made_in", "is_component_of"]
RELS = ["distributed_by", "made_in", "is_component_of"]


def suspicious(v):
    v = v.strip()
    return len(v) > 80 or bool(re.search(r"\b(is|are|was|the text|not (stated|mentioned))\b.*\.", v.lower()))


def parse_log(path):
    doms, reasons = set(), Counter()
    if not os.path.exists(path):
        return doms, reasons
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                try:
                    r = json.loads(line)
                    doms.add(r.get("domain", ""))
                    reasons[str(r.get("error", "")).split(":")[0][:40]] += 1
                    continue
                except json.JSONDecodeError:
                    pass
            m = re.search(r"domain=(\S+)", line)
            if m:
                doms.add(m.group(1))
                rm = re.search(r"reason=(.*)", line)
                reasons[(rm.group(1) if rm else "?")[:40]] += 1
    return doms, reasons


def main():
    os.makedirs(OUT, exist_ok=True)
    universe = set()
    with open(PAGES, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") == "found" and r.get("text"):
                universe.add(r["domain"])
    print(f"[INFO] {len(universe):,} fetched pages with text")

    summary = {"universe": len(universe), "configs": {}, "logs": {}}
    missing_by = {}
    for name, path in CONFIGS.items():
        seen, dup, bad, status = set(), 0, 0, Counter()
        miss_field, nonstr, empty, bad_manu, susp = Counter(), Counter(), Counter(), Counter(), Counter()
        invalid = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    continue
                dom = str(r.get("domain", "")).strip()
                if dom in seen:
                    dup += 1
                    continue
                seen.add(dom)
                status[r.get("status", "<none>")] += 1
                problems = []
                for k in REQUIRED:
                    if k not in r:
                        miss_field[k] += 1
                        problems.append(f"missing:{k}")
                        continue
                    v = r[k]
                    if not isinstance(v, str):
                        nonstr[k] += 1
                        problems.append(f"nonstring:{k}")
                        continue
                    if not v.strip():
                        empty[k] += 1
                        problems.append(f"empty:{k}")
                if isinstance(r.get("manufactures"), str) and \
                        r["manufactures"].strip().lower() not in {"yes", "no", "unknown"}:
                    bad_manu[r["manufactures"].strip().lower()[:30]] += 1
                    problems.append("manufactures_value")
                for k in RELS + ["product", "company"]:
                    if isinstance(r.get(k), str) and suspicious(r[k]):
                        susp[k] += 1
                        problems.append(f"suspicious:{k}")
                if problems:
                    invalid.append({"domain": dom, "problems": ";".join(problems),
                                    **{k: str(r.get(k, ""))[:200] for k in REQUIRED}})
        missing = universe - seen
        extra = seen - universe
        missing_by[name] = missing
        with open(f"{OUT}/audit_missing_{name}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(missing)))
        if invalid:
            with open(f"{OUT}/audit_invalid_{name}.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(invalid[0].keys()))
                w.writeheader()
                w.writerows(invalid)
        s = {"records": len(seen), "unparseable_lines": bad, "duplicate_domains": dup,
             "status": dict(status), "missing_pages": len(missing),
             "records_not_in_fetched_pages": len(extra),
             "missing_field": dict(miss_field), "non_string": dict(nonstr),
             "empty_string": dict(empty), "manufactures_invalid_values": dict(bad_manu.most_common(10)),
             "suspicious_values": dict(susp), "records_with_any_problem": len(invalid)}
        summary["configs"][name] = s
        print(f"\n== {name}: {len(seen):,} records | missing {len(missing):,} | unparseable {bad} "
              f"| duplicates {dup} | with problems {len(invalid):,}")
        print(f"   status {dict(status)}")
        print(f"   missing fields {dict(miss_field)} | non-string {dict(nonstr)} | empty {dict(empty)}")
        print(f"   manufactures not yes/no/unknown: {sum(bad_manu.values())} {dict(bad_manu.most_common(5))}")
        print(f"   suspicious long/sentence values: {dict(susp)}")

    for label, path in LOGS.items():
        doms, reasons = parse_log(path)
        info = {"path": path, "exists": os.path.exists(path), "domains": len(doms),
                "reasons": dict(reasons.most_common(10))}
        for name, miss in missing_by.items():
            info[f"overlap_with_missing_{name}"] = len(doms & miss)
        summary["logs"][label] = info
        print(f"\n== log {label}: {len(doms)} domains, reasons {dict(reasons.most_common(5))}")
        print("   overlap with missing pages: " +
              ", ".join(f"{n}={info[f'overlap_with_missing_{n}']}" for n in missing_by))

    with open(f"{OUT}/audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[DONE] {OUT}/audit_summary.json, audit_missing_*.txt, audit_invalid_*.csv")
    print("NOTE: inside iterative records, a relation call that failed was stored as 'unknown'; "
          "this cannot be detected from the saved files.")


if __name__ == "__main__":
    main()
