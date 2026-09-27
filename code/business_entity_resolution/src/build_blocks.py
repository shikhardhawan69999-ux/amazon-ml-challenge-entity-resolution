import duckdb
import pandas as pd
import time
from src import config

def run_duckdb_blocking(s1_parquet_path, s2s3_parquet_path):
    print("Initializing DuckDB for Out-Of-Core Multi-Blocking...")
    
    # Configure DuckDB to use max 12GB RAM and spill to disk
    con = duckdb.connect(database=':memory:')
    con.execute("PRAGMA memory_limit='12GB'")
    con.execute("PRAGMA threads=8")
    
    # Register the parquet files as views
    con.execute(f"CREATE VIEW s1 AS SELECT * FROM read_parquet('{s1_parquet_path}')")
    con.execute(f"CREATE VIEW s2s3 AS SELECT * FROM read_parquet('{s2s3_parquet_path}')")
    
    print("Executing Multi-Blocking Rules (DuckDB Engine)...")
    t0 = time.time()
    
    # ---------------------------------------------------------
    # MULTI-BLOCKING QUERY
    # We use UNION to combine candidates from different strict filters
    # DuckDB will execute this on 12.5M rows in seconds/minutes!
    # ---------------------------------------------------------
    query = """
    CREATE TABLE candidates AS
    SELECT DISTINCT source1_entity_id, candidate_entity_id FROM (
        
        -- BLOCK 1: Exact Name Match
        SELECT s1.entity_id as source1_entity_id, s2s3.entity_id as candidate_entity_id
        FROM s1 JOIN s2s3 ON s1.business_name_clean = s2s3.business_name_clean
        WHERE s1.business_name_clean != ''
        
        UNION ALL
        
        -- BLOCK 2: Exact Core Name Match
        SELECT s1.entity_id as source1_entity_id, s2s3.entity_id as candidate_entity_id
        FROM s1 JOIN s2s3 ON s1.business_name_core = s2s3.business_name_core
        WHERE s1.business_name_core != ''
        
        UNION ALL
        
        -- BLOCK 3: Pincode + High Jaro-Winkler Similarity on Core Name
        SELECT s1.entity_id as source1_entity_id, s2s3.entity_id as candidate_entity_id
        FROM s1 JOIN s2s3 ON s1.pincode = s2s3.pincode
        WHERE s1.pincode != '' 
          AND jaro_winkler_similarity(s1.business_name_core, s2s3.business_name_core) > 0.85
          
        UNION ALL
        
        -- BLOCK 4: Clean Address Match (Strong Signal)
        SELECT s1.entity_id as source1_entity_id, s2s3.entity_id as candidate_entity_id
        FROM s1 JOIN s2s3 ON s1.business_address_clean = s2s3.business_address_clean
        WHERE s1.business_address_clean != ''
    )
    """
    
    con.execute(query)
    print(f"DuckDB Blocking completed in {time.time() - t0:.2f} seconds!")
    
    print("Extracting candidate pairs to Pandas...")
    candidates = con.execute("SELECT * FROM candidates").df()
    
    # We add a dummy blocking_score for compatibility with existing features
    candidates['blocking_score'] = 1.0 
    
    con.close()
    return candidates
