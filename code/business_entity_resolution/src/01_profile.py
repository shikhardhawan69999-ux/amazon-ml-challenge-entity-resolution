import polars as pl
import json
import os
from src import config

def profile_data():
    print("Starting STEP 1: Data Profiling...")
    
    output_dir = f"{config.BASE_DIR}/code/business_entity_resolution/artifacts"
    os.makedirs(output_dir, exist_ok=True)
    
    stats = {}
    
    # Files to profile
    files = {
        "S1": f"{config.TRAIN_DIR}/train_source1.tsv",
        "S2": f"{config.TRAIN_DIR}/train_source2.tsv",
        "S3": f"{config.TRAIN_DIR}/train_source3.tsv",
    }
    
    for name, filepath in files.items():
        print(f"Profiling {name}...")
        df = pl.scan_csv(filepath, separator="\t", ignore_errors=True)
        
        # Calculate stats lazily, then collect once
        res = df.select([
            pl.len().alias("total_rows"),
            pl.col("business_name").is_null().sum().alias("missing_name"),
            pl.col("business_address").is_null().sum().alias("missing_address"),
            pl.col("country").is_null().sum().alias("missing_country"),
            pl.col("business_name").n_unique().alias("unique_names"),
            pl.col("business_address").n_unique().alias("unique_addresses"),
            pl.col("business_name").str.len_bytes().mean().alias("avg_name_length"),
            pl.col("business_address").str.len_bytes().mean().alias("avg_address_length")
        ]).collect()
        
        row = res.row(0)
        
        stats[name] = {
            "total_rows": row[0],
            "missing_name": row[1],
            "missing_address": row[2],
            "missing_country": row[3],
            "duplicate_names": row[0] - row[4],
            "duplicate_addresses": row[0] - row[5],
            "avg_name_length": round(row[6] or 0, 2),
            "avg_address_length": round(row[7] or 0, 2)
        }
        
    print("Profiling Ground Truth...")
    gt = pl.scan_csv(f"{config.TRAIN_DIR}/train_ground_truth.tsv", separator="\t").collect()
    
    # Calculate matches per S1
    matches_count = (
        gt.with_columns([
            pl.col("matched_entity_ids").fill_null("")
            .map_elements(lambda x: 0 if x == "" else len(x.split(",")), return_dtype=pl.Int64)
            .alias("match_count")
        ])
    )
    
    gt_stats = {
        "s1_with_0_matches": matches_count.filter(pl.col("match_count") == 0).height,
        "s1_with_1_match": matches_count.filter(pl.col("match_count") == 1).height,
        "s1_with_multiple_matches": matches_count.filter(pl.col("match_count") > 1).height,
    }
    
    stats["GroundTruth"] = gt_stats
    
    profile_path = f"{output_dir}/profile.json"
    with open(profile_path, "w") as f:
        json.dump(stats, f, indent=4)
        
    print(f"✅ Profiling complete! Results saved to {profile_path}")
    print(json.dumps(stats, indent=4))

if __name__ == "__main__":
    profile_data()
