"""
Compare raw (naive, full) text against trafilatura's cleaned text for the
same set of domains we've been testing - to see exactly what's being fed
to the LLM vs what the full page actually contained.
"""
import json
import gzip
import io
import csv

import boto3
from bs4 import BeautifulSoup

s3 = boto3.client("s3", region_name="us-east-1")

TARGET_DOMAINS = {"hobbyboden.dk", "feng-shui-architects.com", "titaniumdynamics.com",
                   "deniswhitehomecinema.com.au", "ifairings.com"}


def fetch_and_naive_extract(filename, offset, length):
    byte_range = f"bytes={offset}-{int(offset) + int(length) - 1}"
    resp = s3.get_object(Bucket="commoncrawl", Key=filename, Range=byte_range)
    raw_gz = resp["Body"].read()
    with gzip.GzipFile(fileobj=io.BytesIO(raw_gz)) as gz:
        raw = gz.read()
    _warc, rest = raw.split(b"\r\n\r\n", 1)
    _http, html = rest.split(b"\r\n\r\n", 1)
    html_str = html.decode("utf-8", errors="replace")
    soup = BeautifulSoup(html_str, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)


# Load location info for our target domains
locations = {}
with open("data/interim/athena_lookup_results_all_dedup.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["domain"] in TARGET_DOMAINS:
            locations[row["domain"]] = row

# Load current (trafilatura) text
current = {}
with open("data/interim/fetched_pages.jsonl") as f:
    for line in f:
        r = json.loads(line)
        if r["domain"] in TARGET_DOMAINS:
            current[r["domain"]] = r

for domain in TARGET_DOMAINS:
    print(f"\n{'='*70}")
    print(f"DOMAIN: {domain}")
    print(f"{'='*70}")

    if domain in current:
        traf_text = current[domain]["text"]
        method = current[domain].get("extraction_method", "?")
        print(f"\n--- TRAFILATURA (method={method}, {len(traf_text)} chars) ---")
        print(traf_text[:1000])

    if domain in locations:
        row = locations[domain]
        naive_text = fetch_and_naive_extract(row["warc_filename"], row["warc_record_offset"], row["warc_record_length"])
        print(f"\n--- NAIVE/RAW (full page text, {len(naive_text)} chars) ---")
        print(naive_text[:1000])
