"""
Bulk CDX lookup via AWS Athena, querying Common Crawl's columnar index
directly instead of the rate-limited CDX API. This is a one-time bulk
operation and is NOT subject to index.commoncrawl.org's rate limiting at
all, since it queries S3 data via AWS's own infrastructure (Athena/Glue),
completely separate from the CDX API server.

Usage (from the project root, AWS credentials configured):
    ATHENA_BUCKET=<your-bucket> python -m src.fetch.athena_lookup \
        data/interim/domain_urls.jsonl data/interim/athena_lookup_results_all.csv

Prerequisite: the Common Crawl index table `ccindex` must exist in the Athena
database (see Common Crawl's documentation of its columnar URL index).

<input_jsonl> should have {"domain": ..., "url": ...} per line (same
format as domain_urls.jsonl / domain_urls_remaining.jsonl).

<output_csv> will contain: domain,url,warc_filename,warc_record_offset,
warc_record_length,fetch_time  - one row per URL FOUND in the index
(URLs not found simply won't appear in the output).
"""

import os
import boto3
import json
import csv
import time
import sys
from pathlib import Path

AWS_REGION = "us-east-1"
BUCKET = os.environ.get("ATHENA_BUCKET", "faezeh-thesis-athena-v1")  # S3 bucket for Athena input/results
ATHENA_OUTPUT = f"s3://{BUCKET}/results/"
INPUT_S3_PREFIX = "athena-input/"
INPUT_S3_KEY = f"{INPUT_S3_PREFIX}domain_urls.csv"
DATABASE = "default"

session = boto3.Session(region_name=AWS_REGION)
athena = session.client("athena")
s3 = session.client("s3")


def run_query(sql, database=DATABASE):
    print(f"Running query...\n{sql[:200]}{'...' if len(sql) > 200 else ''}")
    resp = athena.start_query_execution(
        QueryString=sql,
        QueryExecutionContext={"Database": database},
        ResultConfiguration={"OutputLocation": ATHENA_OUTPUT},
    )
    qid = resp["QueryExecutionId"]
    while True:
        status = athena.get_query_execution(QueryExecutionId=qid)["QueryExecution"]["Status"]
        state = status["State"]
        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break
        time.sleep(2)
    if state != "SUCCEEDED":
        reason = status.get("StateChangeReason", "unknown")
        raise RuntimeError(f"Query {state}: {reason}")
    print(f"Query succeeded: {qid}")
    return qid


def prepare_and_upload_csv(input_jsonl_path, s3_key):
    local_csv = Path("/tmp/athena_input.csv")
    count = 0
    with open(input_jsonl_path, encoding="utf-8") as fin, \
         open(local_csv, "w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout, quoting=csv.QUOTE_ALL)
        writer.writerow(["domain", "url"])
        for line in fin:
            record = json.loads(line)
            writer.writerow([record["domain"], record["url"]])
            count += 1
    s3.upload_file(str(local_csv), BUCKET, s3_key)
    print(f"Uploaded {count:,} domains from {input_jsonl_path} -> s3://{BUCKET}/{s3_key}")
    return count


def create_urls_table():
    sql = f"""
    CREATE EXTERNAL TABLE IF NOT EXISTS my_urls (
      domain STRING,
      url STRING
    )
    ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
    WITH SERDEPROPERTIES (
       'separatorChar' = ',',
       'quoteChar' = '"'
    )
    STORED AS TEXTFILE
    LOCATION 's3://{BUCKET}/{INPUT_S3_PREFIX}'
    TBLPROPERTIES ('skip.header.line.count'='1')
    """
    run_query(sql)


def run_join_query():
    sql = """
    SELECT m.domain, m.url, c.warc_filename, c.warc_record_offset,
           c.warc_record_length, c.fetch_time
    FROM my_urls m
    JOIN ccindex c
      ON regexp_replace(m.url, '/$', '') = regexp_replace(c.url, '/$', '')
    WHERE c.crawl = 'CC-MAIN-2020-40'
      AND c.subset = 'warc'
      AND c.fetch_status = 200
    """
    return run_query(sql)


def download_results(qid, output_path):
    key = f"results/{qid}.csv"
    s3.download_file(BUCKET, key, output_path)
    print(f"Downloaded results to {output_path}")


if __name__ == "__main__":
    input_jsonl = sys.argv[1] if len(sys.argv) > 1 else "data/interim/domain_urls_scattered1k.jsonl"
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "data/interim/athena_lookup_results_test.csv"

    total = prepare_and_upload_csv(input_jsonl, INPUT_S3_KEY)
    create_urls_table()
    qid = run_join_query()
    download_results(qid, output_csv)

    with open(output_csv, encoding="utf-8") as f:
        matched = sum(1 for _ in f) - 1  # minus header
    print(f"\n--- Done ---")
    print(f"Requested: {total:,}")
    print(f"Matched:   {matched:,}")
