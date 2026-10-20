# Automated Construction of Supply Chain Knowledge Graphs from Web Crawl Data Using LLMs

Master's thesis, Leuphana University Lüneburg. Supervisor: Dr. Debayan Banerjee.

This repository contains the code for a pipeline that turns web-scale e-commerce pages into a queryable supply chain knowledge graph. It selects one page per domain from the WDC PDC2020-C corpus, retrieves the page text from Common Crawl, extracts entities and relations with an LLM, normalizes and merges them into a graph, and evaluates the result against a human-annotated gold standard and a set of competency questions.

The extraction design is a 2×2 factorial experiment: two models (Qwen3-30B-A3B, Llama 3.3 70B) × two prompting strategies (single-pass, iterative). The prompting follows the schema-guided approach of Brinkmann et al. ([arXiv:2306.14921](https://arxiv.org/abs/2306.14921)). The iterative variant adapts Carta et al. ([arXiv:2307.01128](https://arxiv.org/abs/2307.01128)) by iterating per relation type instead of per entity.

## Pipeline

| Stage | Purpose | Module | Input | Output |
| ----- | ------- | ------ | ----- | ------ |
| A. Domain deduplication | One record per domain (first occurrence), 280,239 domains from 21.9M records | `TODO: path` | WDC corpus | `TODO` |
| B. Page retrieval | Common Crawl index lookup (Athena) + WARC byte-range fetch (S3), text via trafilatura | `TODO: path` | domain list | `data/interim/fetched_pages.jsonl` |
| C. LLM extraction | Entities (company, product, location) and relations, single-pass and iterative | `src/extraction/run_extraction.py` | `fetched_pages.jsonl` | `extraction_results_*.jsonl` |
| Normalization | Entity and location alias normalization | `src/normalization/normalize_entities.py`, `src/normalization/location_aliases.py` | extraction results | `extraction_results_normalized_*.jsonl` |
| D. Graph construction | Build the KG, merge products by identity rules | `src/kg/build_graph.py` | normalized results | `kg_graph_*.gpickle` |
| Competency questions | CQs run against the graph | `src/kg/run_cqs_llama.py` | graph | CQ results |

## Schema

Entity types: `company`, `product`, `location`.

Relations: `manufactures`, `is_component_of`, `distributed_by`, `made_in`. Additionally, `offers` (operator → product) is derived from page context and kept separate from the four LLM-extracted relations.

`TODO: update the relation names here if `distributed_by` is renamed.`

## Requirements

- Python 3.10+
- A GPU server for local inference (developed on 5× NVIDIA L40S, one GPU used at a time)
- vLLM for local serving of Qwen3-30B-A3B-Instruct-2507-FP8, or an OpenAI-compatible API (GWDG AcademicCloud `chat-ai.academiccloud.de`) for Llama 3.3 70B
- AWS credentials with Athena and S3 read access (stage B)
- `TODO: list in requirements.txt / pyproject.toml`

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env     # then fill in your keys
```

Keys are read from `.env`, which is gitignored. Never commit it.

## Running

Run modules from the project root with `-m`, not as plain scripts, because the `src` packages use absolute imports (running `python src/kg/build_graph.py` breaks the location alias import):

```bash
python -m src.extraction.run_extraction     # stage C
python -m src.normalization.normalize_entities
python -m src.kg.build_graph                # stage D
python -m src.kg.run_cqs_llama              # competency questions
```

`TODO: add the arguments for choosing model and strategy (single-pass / iterative).`

## Evaluation

- **Extraction quality:** per-field precision, recall and F1 against a human-annotated gold standard of 320 stratified pages (`gold_annotation.xlsx`).
- **Graph quality:** competency questions executed against the graph, plus an audit of component edges, merged identifiers and name variants.
- **Baseline:** classical supplier-buyer extraction after Wichmann et al.

## Repository structure

```
.
├── src/
│   ├── extraction/        # LLM extraction (single-pass and iterative)
│   ├── normalization/     # entity and location normalization
│   └── kg/                # graph construction and competency questions
├── gold_annotation.xlsx   # final gold standard (320 pages)
├── .env.example
├── requirements.txt
└── README.md
```

`TODO: adjust to the real tree after cleanup.`

## Data

Raw and intermediate data (Common Crawl pages, extraction outputs, graph files) are **not** in this repository because of their size. They are produced by running the stages above. Only the final gold standard is included.

## Reproducibility

`TODO: add run manifest + prompt version hashes once implemented.`

## References

- Brinkmann et al., schema-guided LLM prompting, arXiv:2306.14921
- Carta et al., iterative LLM-based KG construction, arXiv:2307.01128
- Wichmann et al., supplier-buyer relation extraction (baseline)
- WDC PDC2020-C corpus (Web Data Commons)
