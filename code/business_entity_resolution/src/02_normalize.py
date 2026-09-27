import polars as pl
import os
import time
import unicodedata
from src import config

def unicode_normalize(text):
    if text is None:
        return ""
    return unicodedata.normalize("NFKC", str(text))

def run_smoke_test(input_file):
    print(f"\n--- SMOKE TEST ON 50,000 ROWS: {input_file} ---")
    df = pl.scan_csv(
        input_file,
        separator="\t",
        infer_schema_length=5000,
        ignore_errors=True,
        n_rows=50000
    )
    
    # Basic Normalization
    df = df.with_columns([
        pl.col("business_name").fill_null("").cast(pl.String),
        pl.col("business_address").fill_null("").cast(pl.String)
    ])
    
    df = df.with_columns([
        pl.col("business_name")
        .map_elements(lambda x: unicodedata.normalize("NFKC", x), return_dtype=pl.String)
        .str.to_lowercase()
        .str.replace_all(r"&", " and ")
        .str.replace_all(r"[^a-z0-9\s]", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .alias("name_norm"),

        pl.col("business_address")
        .map_elements(lambda x: unicodedata.normalize("NFKC", x), return_dtype=pl.String)
        .str.to_lowercase()
        .str.replace_all(r"&", " and ")
        .str.replace_all(r"[^a-z0-9\s]", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .alias("address_norm")
    ])
    
    final_cols = ["entity_id", "business_name", "business_address", "country", "name_norm", "address_norm"]
    df = df.select([pl.col(c) for c in final_cols if c in df.collect_schema().names() or c in ["name_norm", "address_norm"]])
    
    res = df.collect()
    print("Shape:", res.shape)
    print("Columns:", res.columns)
    print("First 5 rows:")
    print(res.head(5))
    print("Null counts:")
    print(res.null_count())
    print("Smoke test successful!")

def process_file(input_file, output_file):
    start_time = time.time()
    print(f"\nProcessing {input_file}...")
    
    df = pl.scan_csv(
        input_file,
        separator="\t",
        infer_schema_length=5000,
        ignore_errors=True
    )
    
    df = df.with_columns([
        pl.col("business_name").fill_null("").cast(pl.String),
        pl.col("business_address").fill_null("").cast(pl.String)
    ])
    
    df = df.with_columns([
        pl.col("business_name")
        .map_elements(lambda x: unicodedata.normalize("NFKC", x), return_dtype=pl.String)
        .str.to_lowercase()
        .str.replace_all(r"&", " and ")
        .str.replace_all(r"[^a-z0-9\s]", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .alias("name_norm"),

        pl.col("business_address")
        .map_elements(lambda x: unicodedata.normalize("NFKC", x), return_dtype=pl.String)
        .str.to_lowercase()
        .str.replace_all(r"&", " and ")
        .str.replace_all(r"[^a-z0-9\s]", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .alias("address_norm")
    ])
    
    final_cols = ["entity_id", "business_name", "business_address", "country", "name_norm", "address_norm"]
    df = df.select([pl.col(c) for c in final_cols if c in df.collect_schema().names() or c in ["name_norm", "address_norm"]])
    
    # Use streaming execution
    df.sink_parquet(output_file)
    
    # Get stats
    elapsed = time.time() - start_time
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    # Read rows from output parquet efficiently
    num_rows = pl.scan_parquet(output_file).select(pl.len()).collect().item()
    
    print(f"Output File: {output_file}")
    print(f"Rows Processed: {num_rows}")
    print(f"File Size: {file_size_mb:.2f} MB")
    print(f"Elapsed Time: {elapsed:.2f} seconds")

if __name__ == "__main__":
    out_dir = f"{config.BASE_DIR}/code/business_entity_resolution/artifacts/normalized"
    os.makedirs(out_dir, exist_ok=True)
    
    sources = [
        (f"{config.TRAIN_DIR}/train_source1.tsv", f"{out_dir}/s1.parquet"),
        (f"{config.TRAIN_DIR}/train_source2.tsv", f"{out_dir}/s2.parquet"),
        (f"{config.TRAIN_DIR}/train_source3.tsv", f"{out_dir}/s3.parquet")
    ]
    
    # Run smoke test on S1 first
    run_smoke_test(sources[0][0])
    
    print("\n--- CONTINUING WITH FULL DATASET ---")
    for input_file, output_file in sources:
        if os.path.exists(input_file):
            process_file(input_file, output_file)
        else:
            print(f"Skipping {input_file}, file not found.")
            
    print("\nSTEP 2 COMPLETE")
