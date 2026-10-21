# Automated Construction of Supply Chain Knowledge Graphs from Web Crawl Data Using Large Language Models


The pipeline selects one product page per website from the Web Data Commons Product Data Corpus (PDC2020-C), retrieves the archived pages from Common Crawl, extracts company–product information with zero-shot, schema-guided LLM prompting, and builds one knowledge graph per extraction configuration. The graphs are evaluated against a 320-page gold standard, a classical baseline, SPARQL competency questions and sampled graph audits.

The extraction is a 2 × 2 design: two self-hosted models (Qwen3-30B-A3B-Instruct-2507, Llama 3.3 70B Instruct) × two prompting strategies (single-pass, iterative per relation type, adapted from Carta et al. 2023).

---

## Pipeline

| Stage | What it does | Module(s) | Main output |
|---|---|---|---|
| **A** Source selection | 21.9 M WDC product records → one URL per domain (280,239 domains) | `src/fetch/build_domain_list.py` | `domain_urls.jsonl` |
| **B** Retrieval | Common Crawl index lookup with AWS Athena, WARC byte-range reads from S3, text via trafilatura with an 800-character BeautifulSoup fallback (280,238 pages) | `src/fetch/athena_lookup.py` → `dedupe_athena_results.py` → `extract_text_hybrid.py` → `repair_marginal_extractions.py` | `fetched_pages.jsonl` |
| | Re-attach WDC cluster id and declared brand (identity evidence for Stage D) | `src/kg/join_wdc_fields.py` | `wdc_fields.jsonl` |
| **C** LLM extraction | Single-pass (1 call/page) and iterative (up to 5 calls/page) | `src/extraction/run_extraction.py`, `src/extraction/run_iterative_extraction.py` | `extraction_results_*.jsonl` |
| | Schema validation and failure audit of the four runs | `src/kg/audit_outputs.py` | `eval/audit_summary.json` |
| **D** Normalization | Company/brand cleanup and fuzzy merging (threshold 97), location aliases | `src/normalization/normalize_entities.py`, `location_aliases.py` | `extraction_results_*_normalized.jsonl` |
| | Knowledge graphs with evidence-based product identity, competency questions | `src/kg/build_kg.py` | `kg/kg_<config>_<full\|common>.gpickle`, `kg/summary.csv` |
| | RDF export and SPARQL competency questions CQ1–CQ8 | `src/kg/export_rdf.py` | `kg/*.nt`, `kg/sparql/*.rq`, `kg/sparql_*.txt` |
| **Eval** | Stratified gold sample, scoring, baseline, pre-filter analysis, graph audits | `src/kg/make_stratified_annotation_sheet.py`, `score_unified.py`, `classical_baselines.py`, `feature_analysis.py`, `make_audit_sheets.py`, `recheck_cq8.py` | `eval/scores_long.csv`, `kg/audit_*_checked.xlsx` |
| | All numbers and figures used in Chapters 3–5 | `src/kg/collect_thesis_numbers.py` | `eval/thesis_numbers.txt`, `eval/figures/` |

All paths in the Output column are relative to `data/interim/`.


---

## Schema

| Relation | Domain → Range | Meaning |
|---|---|---|
| `manufactures` | company → product | The website operator makes the product (edge only for "yes") |
| `has_brand` | product → company | A different company or brand that makes or brands the product |
| `made_in` | product → location | Stated manufacturing location (not an address or shipping origin) |
| `is_component_of` | product → product | One specific, named parent product |
| `offers` | company → product | Derived from page context, not extracted by the LLM |

> **Naming note.** `has_brand` is stored as **`distributed_by`** in the prompts and in all extraction files, because that was its name when the runs were made. It captures the product's brand or named maker, not a distributor, so the thesis, the graphs and the RDF export call it `has_brand` / `sc:hasBrand`.

---

## Requirements

- Python 3.12 (developed with 3.12.3) on Linux
- **Stage B:** an AWS account with Athena and S3 access, and the Common Crawl index table `ccindex` set up in Athena (see Common Crawl's documentation of its columnar URL index)
- **Stage C:** a GPU with vLLM for the self-hosted models (the thesis used one NVIDIA L40S, 46 GB, per run), or any OpenAI-compatible API
- **Stage A input:** the WDC PDC2020-C corpus as JSON lines at `data/raw/pdc2020c.json` (about 15 GB), from [Web Data Commons](https://webdatacommons.org)

Stages D and Eval need neither a GPU nor AWS.

## Installation

```bash
git clone https://github.com/faeze1991/sc_kg_thesis-.git
cd sc_kg_thesis-
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              
set -a; source .env; set +a       
```

`.env` is gitignored and must never be committed.

| Variable | Used by | Purpose |
|---|---|---|
| `EXTRACTION_ENDPOINT` | Stage C | OpenAI-compatible chat-completions URL (vLLM or GWDG) |
| `GWDG_API_KEY` | Stage C | Only for the GWDG AcademicCloud API |
| `EXTRACTION_WORKERS` | Stage C | Concurrent requests (20 for vLLM, 3 for the GWDG API) |
| `AWS_*`, `ATHENA_BUCKET` | Stage B | Athena index lookup and S3 reads |

---

## Running the pipeline

Always run from the project root in module mode (`python -m src....`). The modules use absolute imports such as `from src.normalization.location_aliases import ...`, which fail when a file is run directly as a script.

### Stages A–B: source selection and retrieval

```bash
python -m src.fetch.build_domain_list                     # Stage A: one URL per domain from the WDC corpus
python -m src.fetch.athena_lookup data/interim/domain_urls.jsonl data/interim/athena_lookup_results_all.csv
python -m src.fetch.dedupe_athena_results data/interim/athena_lookup_results_all.csv data/interim/athena_lookup_results_all_dedup.csv
python -m src.fetch.extract_text_hybrid                   # fetch pages from S3 and extract text
python -m src.fetch.repair_marginal_extractions           # re-extracts short pages (needed only because an early run used a lower threshold)
mv data/interim/fetched_pages_repaired.jsonl data/interim/fetched_pages.jsonl
python -m src.kg.join_wdc_fields                          # adds WDC product cluster and brand to each page
```

### Stage C: extraction (four configurations)

Serve a model with vLLM, for example `vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 --port 8000`, then:

```bash
P=data/interim/fetched_pages.jsonl; I=data/interim
QWEN=Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
LLAMA=lambdalabs/Llama-3.3-70B-Instruct-AWQ-4bit

# single-pass
EXTRACTION_ENDPOINT=http://localhost:8000/v1/chat/completions \
  python -m src.extraction.run_extraction $QWEN  $P $I/extraction_results_full_qwen.jsonl     $I/extraction_skipped_full_qwen.log
EXTRACTION_ENDPOINT=http://localhost:8003/v1/chat/completions \
  python -m src.extraction.run_extraction $LLAMA $P $I/extraction_results_llama70b_full.jsonl $I/extraction_skipped_llama70b_full.log

# iterative
EXTRACTION_ENDPOINT=http://localhost:8000/v1/chat/completions \
  python -m src.extraction.run_iterative_extraction $P $I/extraction_results_iterative_full.jsonl       $I/extraction_skipped_iterative_full.log $QWEN
EXTRACTION_ENDPOINT=http://localhost:8003/v1/chat/completions \
  python -m src.extraction.run_iterative_extraction $P $I/extraction_results_iterative_llama_full.jsonl $I/extraction_skipped_iterative_llama_full.log $LLAMA

python -m src.kg.audit_outputs                             # coverage and failure audit (Table 5.1)
```

All runs use temperature 0 and the first 3,000 characters of each page. They can be resumed: pages already in the output file are skipped.

### Stage D: normalization and knowledge graphs

```bash
I=data/interim
python -m src.normalization.normalize_entities $I/extraction_results_full_qwen.jsonl            $I/extraction_results_normalized_FINAL.jsonl
python -m src.normalization.normalize_entities $I/extraction_results_iterative_full.jsonl       $I/extraction_results_iterative_normalized.jsonl
python -m src.normalization.normalize_entities $I/extraction_results_llama70b_full.jsonl        $I/extraction_results_llama70b_normalized.jsonl
python -m src.normalization.normalize_entities $I/extraction_results_iterative_llama_full.jsonl $I/extraction_results_iterative_llama_normalized.jsonl

python -m src.kg.build_kg          # 4 configurations x {full, common page set}
python -m src.kg.export_rdf        # N-Triples + SPARQL CQ1-CQ8, checked against the Python CQ counts
```

### Evaluation

```bash
python -m src.kg.score_unified                             # LLM configurations vs. the gold standard
python -m src.kg.classical_baselines                       # TF-IDF + SVM / LogReg, trained on silver labels
python -m src.kg.score_unified \
    --extra classical_svm=data/interim/eval/preds_classical_svm.jsonl \
    --extra classical_lr=data/interim/eval/preds_classical_lr.jsonl
python -m src.kg.feature_analysis                          # RQ1: what a pre-filter would have kept or lost
python -m src.kg.make_audit_sheets                         # graph audit sheets (filled in by hand)
python -m src.kg.recheck_cq8 data/interim/kg/audit_sheets_checked.xlsx   # after tightening the identity rules
python -m src.kg.collect_thesis_numbers > data/interim/eval/thesis_numbers.txt 2>&1
```

---

## Data in this repository

Raw and intermediate data (the WDC corpus, page texts, extraction outputs, graph files) are not included because of their size. They are regenerated by the stages above. The repository contains the annotation data and the small result files behind the thesis:

```
data/
├── annotation/
│   ├── gold_annotation.xlsx                    
│   ├── annotation_strat_BLIND.xlsx              # blind annotation sheet with the page text of the 320 pages
│   ├── annotation_strat_design.json             # stratum sizes (for corpus-weighted estimates)
│   ├── annotation_strat_reference_HIDDEN.jsonl  # strata, inclusion probabilities and model predictions per gold page
│   └── silver/                                  # 1,166 pages of earlier rounds, labelled with LLM assistance;
│       ├── annotation_sample_WITH_CLAUDE.csv    #   training data for the classical baseline only,
│       └── annotation_topup_WITH_CLAUDE.csv     #   never used for evaluation (semicolon-separated)
└── interim/
    ├── eval/
    │   ├── scores_long.csv          # every system x gold set x policy x matching mode x field (P/R/F1, 95% CI)
    │   ├── page_details.csv         # per page and field: gold, prediction, outcome (error analysis)
    │   ├── coverage.csv             # gold / scored / missing pages per system
    │   └── audit_summary.json       # schema and failure audit of the four extraction runs
    └── kg/
        ├── summary.csv              # graph statistics per configuration and page set (final rules)
        ├── summary_v1.csv, summary_v2.csv   # statistics before each tightening of the identity rules (Figure 5.4)
        ├── entity_matches_*.csv     # log of every accepted merge, rejected cluster and merge candidate
        ├── sparql/cq*.rq            # the competency-question queries (Appendix A.2)
        ├── audit_sheets_checked.xlsx    # first graph audit with verdicts
        ├── audit_cq8_v2_checked.xlsx    # fresh CQ8 audit on the revised graphs
        └── cq8_recheck.csv          # which audited CQ8 pairs survive the revised identity rules
```


## Repository structure

```
.
├── config/schema.yaml          # extraction schema (documentation)
├── data/                       # see above
├── src/
│   ├── fetch/                  # Stages A-B: domain selection, Athena lookup, WARC retrieval, text extraction
│   ├── extraction/             # Stage C: single-pass and iterative LLM extraction
│   ├── normalization/          # Stage D: company/brand normalization, location aliases
│   └── kg/                     # graph construction, RDF/SPARQL, evaluation, baseline, audits
├── .env.example
├── requirements.txt
└── README.md
```


---

## References

Numbers refer to the thesis bibliography.

- [1] S. Carta et al. (2023). *Iterative Zero-Shot LLM Prompting for Knowledge Graph Construction.* arXiv:2307.01128
- [8] P. Wichmann et al. (2020). *Extracting supply chain maps from news articles using deep neural networks.* International Journal of Production Research 58(17)
- [15] A. Brinkmann et al. (2023). *Product Information Extraction using ChatGPT.* arXiv:2306.14921
- [16] J. Kim et al. (2025). *Extracting Supply Chain Information From News Articles Using Large Language Models: A Fully Automatic Approach.* IEEE Access 13
- [18] R. Peeters, R. C. Der, C. Bizer (2023). *WDC Products: A Multi-Dimensional Entity Matching Benchmark.* arXiv:2301.09521
- Common Crawl, snapshot CC-MAIN-2020-40, https://commoncrawl.org
