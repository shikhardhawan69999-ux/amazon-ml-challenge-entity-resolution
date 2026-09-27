import polars as pl
import os
from src import config

def clean_file_with_polars(input_file, output_file, n_rows=None):
    print(f"Streaming and cleaning {input_file} (Saving to {output_file})...")
    
    # scan_csv builds a lazy computation graph, it doesn't load data into RAM!
    lazy_df = pl.scan_csv(
        input_file,
        separator="\t",
        infer_schema_length=5000,
        ignore_errors=True,
        n_rows=n_rows
    )
    
    # Apply basic cleaning without heavy abbreviation expansion
    clean_df = lazy_df.with_columns([
        pl.col("business_name")
        .cast(pl.String)
        .str.to_lowercase()
        .str.replace_all(r"[^a-z0-9\s]", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .alias("name_clean"),

        pl.col("business_address")
        .cast(pl.String)
        .str.to_lowercase()
        .str.replace_all(r"[^a-z0-9\s]", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .alias("address_clean"),
    ])
    
    # Keep essential columns
    clean_df = clean_df.select([
        "entity_id",
        "business_name",
        "business_address",
        "country",
        "name_clean",
        "address_clean"
    ])
    
    # sink_parquet streams the result chunk-by-chunk to disk! (Zero RAM bloat)
    clean_df.sink_parquet(output_file)
    print(f"DONE! Saved to {output_file}")

if __name__ == "__main__":
    os.makedirs(f"{config.BASE_DIR}/code/business_entity_resolution/cleaned_data", exist_ok=True)
    
    s1_in = f"{config.TRAIN_DIR}/train_source1.tsv"
    s1_out = f"{config.BASE_DIR}/code/business_entity_resolution/cleaned_data/s1_clean.parquet"
    
    s2_in = f"{config.TRAIN_DIR}/train_source2.tsv"
    s2_out = f"{config.BASE_DIR}/code/business_entity_resolution/cleaned_data/s2_clean.parquet"
    
    s3_in = f"{config.TRAIN_DIR}/train_source3.tsv"
    s3_out = f"{config.BASE_DIR}/code/business_entity_resolution/cleaned_data/s3_clean.parquet"
    
    # Test on 50,000 rows first as requested!
    print("--- TESTING ON 50,000 ROWS ---")
    clean_file_with_polars(s1_in, s1_out, n_rows=50000)
    
    print("\nAgar ye error ke bina chal jaye, toh 'n_rows=50000' hata kar S2 aur S3 par bhi chala sakte hain!")
