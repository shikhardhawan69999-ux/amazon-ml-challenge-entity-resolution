import polars as pl
import os
from src import config

def normalize_dataset(input_file, output_file, n_rows=None):
    print(f"Normalizing {input_file} (Saving to {output_file})...")
    
    # 1. Base scan
    df = pl.scan_csv(
        input_file,
        separator="\t",
        infer_schema_length=5000,
        ignore_errors=True,
        n_rows=n_rows
    )
    
    # Fill nulls with empty string to prevent string method errors
    df = df.with_columns([
        pl.col("business_name").fill_null("").cast(pl.String),
        pl.col("business_address").fill_null("").cast(pl.String)
    ])
    
    # 2. Name Normalization
    df = df.with_columns([
        pl.col("business_name")
        .str.to_lowercase()
        .str.replace_all(r"[^a-z0-9\s]", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .alias("name_norm")
    ])
    
    # Helper to remove legal suffixes for name_core
    legal_suffixes = r"\b(pvt|private|ltd|limited|co|company|corp|corporation|inc|incorporated|llc|plc)\b"
    df = df.with_columns([
        pl.col("name_norm")
        .str.replace_all(legal_suffixes, "")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .alias("name_core")
    ])
    
    # Sorted and Compact names
    # Note: sorting tokens in Polars expressions: split -> list.sort -> join
    df = df.with_columns([
        pl.col("name_core").str.split(" ").list.eval(pl.element().sort()).list.join(" ").alias("name_sorted"),
        pl.col("name_core").str.replace_all(r"\s+", "").alias("name_compact")
    ])
    
    # 3. Address Normalization
    df = df.with_columns([
        pl.col("business_address")
        .str.to_lowercase()
        .str.replace_all(r"[^a-z0-9\s]", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .alias("address_norm")
    ])
    
    # Pincode / House Number Extraction using Regex
    df = df.with_columns([
        # Postal Code: 5 or 6 digit standalone number
        pl.col("business_address")
        .str.extract(r"\b(\d{5,6})\b", 1)
        .alias("postal_code"),
        
        # House Number: looks for 'no 123', 'plot 123', or just numbers at start
        pl.col("address_norm")
        .str.extract(r"\b(?:no|plot|flat|f no|sy no|door no)\s*(\d+[a-z]?)\b", 1)
        .alias("house_number")
    ])
    
    # Address Unique (removing duplicates tokens if needed, skipping complex list eval for speed)
    df = df.with_columns([
        pl.col("address_norm").str.split(" ").list.unique().list.join(" ").alias("address_unique")
    ])
    
    # Select columns (Keeping original + normalized as per blueprint)
    final_cols = [
        "entity_id", "business_name", "business_address", "country",
        "name_norm", "name_core", "name_sorted", "name_compact",
        "address_norm", "address_unique", "postal_code", "house_number"
    ]
    
    # Select only the available columns
    available_cols = df.collect_schema().names()
    selected_cols = [c for c in final_cols if c in available_cols]
    
    # Wait, some columns like name_norm were added via alias, they exist in the lazy schema.
    # We can just select them directly.
    df = df.select([pl.col(c) for c in final_cols])
    
    # 4. Stream to Parquet (Disk)
    df.sink_parquet(output_file)
    print(f"DONE! Saved to {output_file}")

if __name__ == "__main__":
    out_dir = f"{config.BASE_DIR}/code/business_entity_resolution/artifacts/normalized"
    os.makedirs(out_dir, exist_ok=True)
    
    s1_in = f"{config.TRAIN_DIR}/train_source1.tsv"
    s1_out = f"{out_dir}/s1.parquet"
    
    s2_in = f"{config.TRAIN_DIR}/train_source2.tsv"
    s2_out = f"{out_dir}/s2.parquet"
    
    s3_in = f"{config.TRAIN_DIR}/train_source3.tsv"
    s3_out = f"{out_dir}/s3.parquet"
    
    # Testing ONLY on 50,000 rows as explicitly requested in Blueprint!
    print("--- PHASE 2: NORMALIZATION (Testing 50k rows on S1) ---")
    normalize_dataset(s1_in, s1_out, n_rows=50000)
    
    print("\nBlueprint Check: S1 50k test successful! Aap 'n_rows=50000' hata kar S1, S2, S3 teeno ko ek saath chala sakte hain.")
