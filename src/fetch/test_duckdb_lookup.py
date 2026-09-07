"""
Test: query Common Crawl's columnar (Parquet) index via DuckDB instead of
the rate-limited CDX API, to find warc_filename/offset/length for our URLs.
"""
import duckdb
import json
import time
from pathlib import Path

SAMPLE_SIZE = 500

urls = []
with open("data/interim/domain_urls.jsonl") as f:
    for i, line in enumerate(f):
        if i >= SAMPLE_SIZE:
            break
        urls.append(json.loads(line)["url"])

print(f"Testing with {len(urls)} URLs...")

con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")
con.execute("SET s3_region='us-east-1';")

con.execute("CREATE TABLE my_urls (url VARCHAR)")
con.executemany("INSERT INTO my_urls VALUES (?)", [(u,) for u in urls])

start = time.time()
result = con.execute("""
    SELECT m.url, c.warc_filename, c.warc_record_offset, c.warc_record_length
    FROM my_urls m
    JOIN read_parquet(
        's3://commoncrawl/cc-index/table/cc-main/warc/crawl=CC-MAIN-2020-40/subset=warc/*.parquet',
        hive_partitioning=1
    ) c
    ON m.url = c.url
    WHERE c.fetch_status = 200
""").fetchall()
elapsed = time.time() - start

print(f"\nMatched {len(result)} / {len(urls)} URLs")
print(f"Query time: {elapsed:.1f}s")
if result:
    print(f"\nSample match: {result[0]}")
