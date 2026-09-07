"""
Narrower test: does authenticated GetObject work at all against a single,
exact (non-glob) Common Crawl file, using explicit env-var credential chain?
"""
import duckdb

con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")
con.execute("INSTALL aws; LOAD aws;")
con.execute("""
    CREATE OR REPLACE SECRET secret (
        TYPE s3,
        PROVIDER credential_chain,
        CHAIN 'env',
        REGION 'us-east-1'
    );
""")

# Test 1: fetch the known-real cc-index table's root listing/info page (single file, no glob)
print("Test 1: fetching a single known small file (index.html)...")
try:
    result = con.execute("""
        SELECT * FROM read_text('s3://commoncrawl/cc-index/table/cc-main/index.html')
    """).fetchall()
    print("SUCCESS - single file GetObject works")
except Exception as e:
    print(f"FAILED: {e}")

# Test 2: try listing just this one small known prefix (minimal glob scope)
print("\nTest 2: attempting to glob just the metadata files at the table root...")
try:
    result = con.execute("""
        SELECT * FROM glob('s3://commoncrawl/cc-index/table/cc-main/warc/crawl=CC-MAIN-2020-40/*')
    """).fetchall()
    print(f"SUCCESS - found {len(result)} items")
    for r in result[:5]:
        print(r)
except Exception as e:
    print(f"FAILED: {e}")
