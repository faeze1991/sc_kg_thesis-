"""
Stage A: Extract one representative URL per unique domain from the WDC PDC2020-C corpus.

Approach: first-occurrence URL per domain (simple, reproducible, consistent with
pilot methodology). No homepage-preference heuristics — kept intentionally simple
per supervisor guidance to avoid over-engineering the retrieval step.

Input:  data/raw/pdc2020c.json          (21.9M product records, JSON-lines)
Output: data/interim/domain_urls.jsonl  (one row per unique domain: {"domain": ..., "url": ...})
        data/interim/stage_a_skipped.log (skipped/malformed line log)

Domain normalization: "www." prefix stripped so www.X.com and X.com are treated
as the same company/domain. Port and userinfo (if present) are also stripped.
"""

import json
from urllib.parse import urlparse
from pathlib import Path

INPUT_PATH = Path("data/raw/pdc2020c.json")
OUTPUT_PATH = Path("data/interim/domain_urls.jsonl")
SKIPPED_LOG_PATH = Path("data/interim/stage_a_skipped.log")


def normalize_domain(url: str):
    """Extract and normalize the domain from a URL. Returns None if invalid."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        if "@" in domain:
            domain = domain.rsplit("@", 1)[-1]

        if ":" in domain:
            domain = domain.split(":", 1)[0]

        if domain.startswith("www."):
            domain = domain[4:]

        return domain if domain else None
    except Exception:
        return None


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    seen_domains = set()
    total_lines = 0
    written = 0
    skipped_null_url = 0
    skipped_malformed_json = 0
    skipped_invalid_domain = 0

    with open(INPUT_PATH, "r", encoding="utf-8") as infile, \
         open(OUTPUT_PATH, "w", encoding="utf-8") as outfile, \
         open(SKIPPED_LOG_PATH, "w", encoding="utf-8") as skiplog:

        for line in infile:
            total_lines += 1

            if total_lines % 1_000_000 == 0:
                print(f"Processed {total_lines:,} lines | {len(seen_domains):,} unique domains so far")

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped_malformed_json += 1
                skiplog.write(f"MALFORMED_JSON\tline={total_lines}\n")
                continue

            if not isinstance(record, dict):
                skipped_malformed_json += 1
                skiplog.write(f"NOT_A_DICT\tline={total_lines}\ttype={type(record).__name__}\n")
                continue

            url = record.get("url")
            if not url:
                skipped_null_url += 1
                skiplog.write(f"NULL_URL\tline={total_lines}\tid={record.get('id')}\n")
                continue

            domain = normalize_domain(url)
            if not domain:
                skipped_invalid_domain += 1
                skiplog.write(f"INVALID_DOMAIN\tline={total_lines}\turl={url}\n")
                continue

            if domain in seen_domains:
                continue

            seen_domains.add(domain)
            outfile.write(json.dumps({"domain": domain, "url": url}) + "\n")
            written += 1

    print("\n--- Stage A complete ---")
    print(f"Total lines processed:     {total_lines:,}")
    print(f"Unique domains written:    {written:,}")
    print(f"Skipped - null url:        {skipped_null_url:,}")
    print(f"Skipped - malformed json:  {skipped_malformed_json:,}")
    print(f"Skipped - invalid domain:  {skipped_invalid_domain:,}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Skip log: {SKIPPED_LOG_PATH}")


if __name__ == "__main__":
    main()
