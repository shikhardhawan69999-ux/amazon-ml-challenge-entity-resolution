import duckdb
import os
import time
from src import config

def run_blocking():
    print("Starting STEP 3: BLOCKING / CANDIDATE GENERATION...")
    
    out_dir = f"{config.BASE_DIR}/code/business_entity_resolution/artifacts/candidates"
    os.makedirs(out_dir, exist_ok=True)
    temp_dir = f"{config.BASE_DIR}/code/business_entity_resolution/artifacts/duckdb_temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Configure DuckDB for high memory limit and temp spilling
    con = duckdb.connect()
    con.execute(f"PRAGMA temp_directory='{temp_dir}';")
    con.execute("PRAGMA memory_limit='12GB';")
    con.execute("PRAGMA threads=8;")
    
    norm_dir = f"{config.BASE_DIR}/code/business_entity_resolution/artifacts/normalized"
    s1_path = f"{norm_dir}/s1.parquet"
    s2_path = f"{norm_dir}/s2.parquet"
    s3_path = f"{norm_dir}/s3.parquet"
    
    if not os.path.exists(s1_path) or not os.path.exists(s2_path) or not os.path.exists(s3_path):
        print("Error: Normalized parquet files not found! Run Step 2 first.")
        return
        
    print("Loading data into DuckDB Views...")
    
    # We dynamically create name_core, name_sorted, and postal_code in DuckDB!
    con.execute(f"""
        CREATE VIEW s1 AS 
        SELECT 
            entity_id, 
            name_norm,
            trim(regexp_replace(name_norm, '\\b(pvt|private|ltd|limited|co|company|corp|corporation|inc|incorporated|llc|plc)\\b', '', 'g')) as name_core,
            array_to_string(list_sort(string_split(trim(regexp_replace(name_norm, '\\b(pvt|private|ltd|limited|co|company|corp|corporation|inc|incorporated|llc|plc)\\b', '', 'g')), ' ')), ' ') as name_sorted,
            regexp_extract(address_norm, '\\b(\\d{{5,6}})\\b', 1) as postal_code
        FROM read_parquet('{s1_path}')
        LIMIT 50000;  -- SMOKE TEST ON FIRST 50,000 ROWS
    """)
    
    con.execute(f"""
        CREATE VIEW s2s3 AS 
        SELECT 
            entity_id, 
            'S2' as candidate_source,
            name_norm,
            trim(regexp_replace(name_norm, '\\b(pvt|private|ltd|limited|co|company|corp|corporation|inc|incorporated|llc|plc)\\b', '', 'g')) as name_core,
            array_to_string(list_sort(string_split(trim(regexp_replace(name_norm, '\\b(pvt|private|ltd|limited|co|company|corp|corporation|inc|incorporated|llc|plc)\\b', '', 'g')), ' ')), ' ') as name_sorted,
            regexp_extract(address_norm, '\\b(\\d{{5,6}})\\b', 1) as postal_code
        FROM read_parquet('{s2_path}')
        UNION ALL
        SELECT 
            entity_id, 
            'S3' as candidate_source,
            name_norm,
            trim(regexp_replace(name_norm, '\\b(pvt|private|ltd|limited|co|company|corp|corporation|inc|incorporated|llc|plc)\\b', '', 'g')) as name_core,
            array_to_string(list_sort(string_split(trim(regexp_replace(name_norm, '\\b(pvt|private|ltd|limited|co|company|corp|corporation|inc|incorporated|llc|plc)\\b', '', 'g')), ' ')), ' ') as name_sorted,
            regexp_extract(address_norm, '\\b(\\d{{5,6}})\\b', 1) as postal_code
        FROM read_parquet('{s3_path}');
    """)
    
    print("Building Rare Token frequencies from S2+S3...")
    # Calculate token frequencies in S2/S3
    con.execute("""
        CREATE TABLE rare_tokens AS
        SELECT token, count(*) as freq
        FROM (
            SELECT unnest(string_split(name_core, ' ')) as token
            FROM s2s3
        )
        WHERE token != '' AND length(token) > 2
        GROUP BY token
        HAVING count(*) BETWEEN 2 AND 100;  -- ONLY VERY RARE TOKENS TO PREVENT EXPLOSION
    """)
    
    # Also index S2S3 tokens for fast join
    con.execute("""
        CREATE TABLE s2s3_tokens AS
        SELECT entity_id, candidate_source, unnest(string_split(name_core, ' ')) as token
        FROM s2s3;
    """)
    # Filter only rare tokens
    con.execute("""
        CREATE TABLE s2s3_rare_tokens AS
        SELECT a.entity_id, a.candidate_source, a.token
        FROM s2s3_tokens a
        JOIN rare_tokens b ON a.token = b.token;
    """)
    
    print("Running Blocking Rules (UNION ALL)...")
    start_time = time.time()
    
    # Run Blocks
    con.execute("""
        CREATE TABLE candidates_raw AS
        
        -- BLOCK 1: name_norm
        SELECT s1.entity_id as source1_entity_id, s2.entity_id as candidate_entity_id, s2.candidate_source
        FROM s1 JOIN s2s3 s2 ON s1.name_norm = s2.name_norm
        WHERE s1.name_norm != ''
        
        UNION ALL
        -- BLOCK 2: name_core
        SELECT s1.entity_id, s2.entity_id, s2.candidate_source
        FROM s1 JOIN s2s3 s2 ON s1.name_core = s2.name_core
        WHERE s1.name_core != ''
        
        UNION ALL
        -- BLOCK 3: name_sorted
        SELECT s1.entity_id, s2.entity_id, s2.candidate_source
        FROM s1 JOIN s2s3 s2 ON s1.name_sorted = s2.name_sorted
        WHERE s1.name_sorted != ''
        
        UNION ALL
        -- BLOCK 4: postal_code
        SELECT s1.entity_id, s2.entity_id, s2.candidate_source
        FROM s1 JOIN s2s3 s2 ON s1.postal_code = s2.postal_code
        WHERE s1.postal_code != '' AND s1.postal_code IS NOT NULL
        
        UNION ALL
        -- BLOCK 5: rare tokens
        SELECT s1_t.entity_id, s2.entity_id, s2.candidate_source
        FROM (
            SELECT entity_id, unnest(string_split(name_core, ' ')) as token FROM s1
        ) s1_t
        JOIN rare_tokens r ON s1_t.token = r.token
        JOIN s2s3_rare_tokens s2 ON s1_t.token = s2.token;
    """)
    
    print("Deduplicating candidates and counting blocks...")
    con.execute(f"""
        CREATE TABLE final_candidates AS
        SELECT 
            source1_entity_id, 
            candidate_entity_id, 
            ANY_VALUE(candidate_source) as candidate_source, 
            COUNT(*) as block_count
        FROM candidates_raw
        GROUP BY source1_entity_id, candidate_entity_id;
    """)
    
    # Save to Parquet
    out_file = f"{out_dir}/candidates_50k.parquet"
    con.execute(f"COPY final_candidates TO '{out_file}' (FORMAT PARQUET);")
    
    elapsed = time.time() - start_time
    
    # Calculate stats
    stats = con.execute("""
        SELECT 
            (SELECT count(*) FROM s1) as s1_rows,
            (SELECT count(*) FROM final_candidates) as total_candidates,
            (SELECT avg(cnt) FROM (SELECT count(*) as cnt FROM final_candidates GROUP BY source1_entity_id)) as avg_cand,
            (SELECT max(cnt) FROM (SELECT count(*) as cnt FROM final_candidates GROUP BY source1_entity_id)) as max_cand,
            (SELECT count(*) FROM s1 WHERE entity_id NOT IN (SELECT source1_entity_id FROM final_candidates)) as zero_cand
    """).fetchone()
    
    print(f"\n--- BLOCKING SMOKE TEST RESULTS ---")
    print(f"S1 Rows Processed: {stats[0]}")
    print(f"Total Candidate Pairs: {stats[1]}")
    print(f"Average Candidates per S1: {stats[2]:.2f}" if stats[2] else "Average Candidates per S1: 0")
    print(f"Max Candidates for one S1: {stats[3]}")
    print(f"S1 Entities with 0 candidates: {stats[4]}")
    
    print("\n--- BLOCK COUNT DISTRIBUTION ---")
    dist = con.execute("SELECT block_count, count(*) FROM final_candidates GROUP BY block_count ORDER BY block_count").fetchall()
    for row in dist:
        print(f"Block Count {row[0]}: {row[1]} pairs")
        
    print(f"\nElapsed Time: {elapsed:.2f} seconds")
    
    # Validation against Ground Truth
    gt_path = f"{config.TRAIN_DIR}/train_ground_truth.tsv"
    if os.path.exists(gt_path):
        print("\n--- VALIDATING BLOCKING RECALL ---")
        con.execute(f"""
            CREATE TABLE gt AS 
            SELECT * FROM read_csv_auto('{gt_path}', sep='\t');
        """)
        
        # Unnest ground truth matched IDs
        con.execute("""
            CREATE TABLE gt_expanded AS
            SELECT source1_entity_id, unnest(string_split(matched_entity_ids, ',')) as true_match_id
            FROM gt
            WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids != '';
        """)
        
        # Keep only GT for the 50k we processed
        recall_stats = con.execute("""
            WITH relevant_gt AS (
                SELECT g.* FROM gt_expanded g
                JOIN s1 ON g.source1_entity_id = s1.entity_id
            ),
            captured AS (
                SELECT r.* FROM relevant_gt r
                JOIN final_candidates c 
                  ON c.source1_entity_id = r.source1_entity_id 
                 AND c.candidate_entity_id = r.true_match_id
            )
            SELECT 
                (SELECT count(*) FROM relevant_gt) as true_pairs,
                (SELECT count(*) FROM captured) as captured_pairs;
        """).fetchone()
        
        true_pairs = recall_stats[0]
        captured_pairs = recall_stats[1]
        
        if true_pairs > 0:
            recall = (captured_pairs / true_pairs) * 100
            print(f"True Matching Pairs (in 50k): {true_pairs}")
            print(f"Captured Matching Pairs: {captured_pairs}")
            print(f"Blocking Recall: {recall:.2f}%")
        else:
            print("No true pairs found in the first 50k rows!")
            
        quantiles = con.execute("""
            SELECT 
                quantile_cont(cnt, 0.95),
                quantile_cont(cnt, 0.99)
            FROM (
                SELECT count(*) as cnt FROM final_candidates GROUP BY source1_entity_id
            )
        """).fetchone()
        print(f"P95 candidates per S1: {quantiles[0]:.0f}" if quantiles[0] else "P95 candidates per S1: N/A")
        print(f"P99 candidates per S1: {quantiles[1]:.0f}" if quantiles[1] else "P99 candidates per S1: N/A")
        
    print("\nSTEP 3 COMPLETE")

if __name__ == "__main__":
    run_blocking()
