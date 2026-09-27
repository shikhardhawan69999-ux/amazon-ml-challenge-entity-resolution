import polars as pl
import os
import time
from rapidfuzz import fuzz
from src import config

def calculate_features():
    print("Starting STEP 4: FEATURE ENGINEERING...")
    
    start_time = time.time()
    
    norm_dir = f"{config.BASE_DIR}/code/business_entity_resolution/artifacts/normalized"
    cand_dir = f"{config.BASE_DIR}/code/business_entity_resolution/artifacts/candidates"
    out_dir = f"{config.BASE_DIR}/code/business_entity_resolution/artifacts/features"
    os.makedirs(out_dir, exist_ok=True)
    
    cand_file = f"{cand_dir}/candidates_50k.parquet"
    if not os.path.exists(cand_file):
        print(f"Error: {cand_file} not found! Run Step 3 first.")
        return
        
    print("Loading Candidates, S1, S2, S3...")
    
    # 1. Load Candidates
    candidates = pl.read_parquet(cand_file)
    print(f"Loaded {candidates.height} candidate pairs.")
    
    # 2. Load Normalized Data
    # For S1, we only need rows that are in candidates to save RAM
    s1_ids = candidates.select("source1_entity_id").unique()
    s1 = pl.scan_parquet(f"{norm_dir}/s1.parquet").join(s1_ids.lazy(), left_on="entity_id", right_on="source1_entity_id", how="inner").collect()
    
    s2_ids = candidates.filter(pl.col("candidate_source") == "S2").select("candidate_entity_id").unique()
    s2 = pl.scan_parquet(f"{norm_dir}/s2.parquet").join(s2_ids.lazy(), left_on="entity_id", right_on="candidate_entity_id", how="inner").collect()
    
    s3_ids = candidates.filter(pl.col("candidate_source") == "S3").select("candidate_entity_id").unique()
    s3 = pl.scan_parquet(f"{norm_dir}/s3.parquet").join(s3_ids.lazy(), left_on="entity_id", right_on="candidate_entity_id", how="inner").collect()
    
    s2s3 = pl.concat([s2, s3])
    
    del s2, s3
    
    print("Creating dynamic normalization features (name_core, postal_code, etc.)...")
    
    # UDF for dynamic features
    legal_suffixes = r"\b(pvt|private|ltd|limited|co|company|corp|corporation|inc|incorporated|llc|plc)\b"
    
    def enrich_data(df):
        return df.with_columns([
            pl.col("name_norm").fill_null(""),
            pl.col("address_norm").fill_null(""),
            pl.col("country").fill_null("").str.to_lowercase()
        ]).with_columns([
            pl.col("name_norm").str.replace_all(legal_suffixes, "").str.replace_all(r"\s+", " ").str.strip_chars().alias("name_core"),
            pl.col("address_norm").str.extract(r"\b(\d{5,6})\b", 1).alias("postal_code"),
            pl.col("address_norm").str.extract(r"\b(?:no|plot|flat|f no|sy no|door no)\s*(\d+[a-z]?)\b", 1).alias("house_number")
        ]).with_columns([
            pl.col("name_core").str.split(" ").list.eval(pl.element().sort()).list.join(" ").alias("name_sorted")
        ])
        
    s1 = enrich_data(s1)
    s2s3 = enrich_data(s2s3)
    
    print("Joining candidates with features...")
    
    # 3. Join candidates with text features
    df = candidates.join(
        s1.rename({c: f"s1_{c}" for c in s1.columns}), 
        left_on="source1_entity_id", right_on="s1_entity_id", how="left"
    ).join(
        s2s3.rename({c: f"s2_{c}" for c in s2s3.columns}),
        left_on="candidate_entity_id", right_on="s2_entity_id", how="left"
    )
    
    del s1, s2s3, candidates
    
    print("Creating Labels from Ground Truth...")
    # 4. Create Labels
    gt = pl.read_csv(f"{config.TRAIN_DIR}/train_ground_truth.tsv", separator="\t")
    # explode ground truth so each S1-S2 is a row
    gt_exploded = gt.with_columns(
        pl.col("matched_entity_ids").fill_null("").str.split(",")
    ).explode("matched_entity_ids")
    
    gt_exploded = gt_exploded.filter(pl.col("matched_entity_ids") != "")
    
    # Add a column "is_match" = 1
    gt_exploded = gt_exploded.with_columns(pl.lit(1).alias("is_match").cast(pl.UInt8))
    
    # Join with df
    df = df.join(
        gt_exploded,
        left_on=["source1_entity_id", "candidate_entity_id"],
        right_on=["source1_entity_id", "matched_entity_ids"],
        how="left"
    )
    
    df = df.with_columns(pl.col("is_match").fill_null(0).alias("label"))
    df = df.drop("is_match")
    
    positive_count = df.filter(pl.col("label") == 1).height
    negative_count = df.filter(pl.col("label") == 0).height
    total_pairs = df.height
    pos_ratio = (positive_count / total_pairs) * 100 if total_pairs > 0 else 0
    
    print(f"Positive pairs: {positive_count}")
    print(f"Hard negative pairs: {negative_count} (Kept all negatives retrieved by blocking)")
    print(f"Total pairs: {total_pairs}")
    print(f"Positive ratio: {pos_ratio:.2f}%")
    
    print("Calculating ML Features using RapidFuzz...")
    
    # 5. Calculate Features
    df = df.with_columns([
        (pl.col("s1_name_norm") == pl.col("s2_name_norm")).cast(pl.UInt8).alias("name_norm_exact"),
        (pl.col("s1_name_core") == pl.col("s2_name_core")).cast(pl.UInt8).alias("name_core_exact"),
        (pl.col("s1_name_sorted") == pl.col("s2_name_sorted")).cast(pl.UInt8).alias("name_sorted_exact"),
        (pl.col("s1_address_norm") == pl.col("s2_address_norm")).cast(pl.UInt8).alias("address_exact"),
        (pl.col("s1_postal_code") == pl.col("s2_postal_code")).cast(pl.UInt8).alias("postal_code_match"),
        (pl.col("s1_country") == pl.col("s2_country")).cast(pl.UInt8).alias("same_country"),
    ])
    
    df = df.with_columns([
        (pl.col("s1_name_norm").str.len_bytes() - pl.col("s2_name_norm").str.len_bytes()).abs().cast(pl.Int16).alias("name_length_difference"),
        (pl.col("s1_address_norm").str.len_bytes() - pl.col("s2_address_norm").str.len_bytes()).abs().cast(pl.Int16).alias("address_length_difference"),
    ])
    
    def calc_ratio(s1, s2):
        return fuzz.ratio(s1, s2) if s1 and s2 else 0.0

    def calc_token_set(s1, s2):
        return fuzz.token_set_ratio(s1, s2) if s1 and s2 else 0.0
        
    def calc_token_sort(s1, s2):
        return fuzz.token_sort_ratio(s1, s2) if s1 and s2 else 0.0

    print(" - name ratios...")
    df = df.with_columns([
        pl.struct(["s1_name_norm", "s2_name_norm"]).map_elements(lambda x: calc_ratio(x["s1_name_norm"], x["s2_name_norm"]), return_dtype=pl.Float32).alias("name_levenshtein_ratio"),
        pl.struct(["s1_name_norm", "s2_name_norm"]).map_elements(lambda x: calc_token_set(x["s1_name_norm"], x["s2_name_norm"]), return_dtype=pl.Float32).alias("name_token_set_ratio"),
        pl.struct(["s1_name_norm", "s2_name_norm"]).map_elements(lambda x: calc_token_sort(x["s1_name_norm"], x["s2_name_norm"]), return_dtype=pl.Float32).alias("name_token_sort_ratio"),
    ])
    
    print(" - address ratios...")
    df = df.with_columns([
        pl.struct(["s1_address_norm", "s2_address_norm"]).map_elements(lambda x: calc_ratio(x["s1_address_norm"], x["s2_address_norm"]), return_dtype=pl.Float32).alias("address_levenshtein_ratio"),
    ])
    
    # Select final schema
    feature_cols = [
        "source1_entity_id", "candidate_entity_id", "candidate_source", "block_count",
        "name_norm_exact", "name_core_exact", "name_sorted_exact", "address_exact",
        "postal_code_match", "same_country",
        "name_length_difference", "address_length_difference",
        "name_levenshtein_ratio", "name_token_set_ratio", "name_token_sort_ratio",
        "address_levenshtein_ratio",
        "label"
    ]
    
    final_df = df.select(feature_cols)
    
    print("\n--- VALIDATION ---")
    missing_features = final_df.null_count()
    duplicate_pairs = final_df.height - final_df.unique(subset=["source1_entity_id", "candidate_entity_id"]).height
    
    print(f"Number of Features: {len(feature_cols) - 4}") # exclude ids, source, label
    print(f"Duplicate pairs: {duplicate_pairs}")
    print(f"Missing values:\n{missing_features}")
    
    out_file = f"{out_dir}/train_features_50k.parquet"
    final_df.write_parquet(out_file)
    
    print(f"\n✅ Features saved to {out_file}")
    print(f"Elapsed Time: {time.time() - start_time:.2f} seconds")
    
    print("\n--- SAMPLE OUTPUT ---")
    print(final_df.head(10))
    
    print("\nSTEP 4 COMPLETE")

if __name__ == "__main__":
    calculate_features()
