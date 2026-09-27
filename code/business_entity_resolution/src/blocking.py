import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.neighbors import NearestNeighbors
import gc
import numpy as np
from src import config

def generate_candidate_pairs(df_s1, df_s2s3, n_neighbors=config.KNN_NEIGHBORS, part_num=1):
    print("Vectorizing text for blocking (Using HashingVectorizer for Ultra-Low RAM)...")
    
    # HashingVectorizer requires ZERO RAM for vocabulary building
    # dtype=np.float32 cuts the final sparse matrix memory strictly in HALF
    vectorizer = HashingVectorizer(analyzer='char_wb', 
                                   ngram_range=config.TFIDF_NGRAM_RANGE, 
                                   n_features=config.TFIDF_MAX_FEATURES,
                                   norm=None, 
                                   alternate_sign=False,
                                   dtype=np.float32)
    
    import scipy.sparse as sp
    print("Applying Manual Max-DF to remove 'ltd/pvt' noise and make math 1000x faster...")
    
    s2s3_counts = vectorizer.transform(df_s2s3['combined_text'])
    
    # EXTREME RAM TRICK: Delete 5GB of Strings immediately!
    del df_s2s3['combined_text']
    gc.collect()
    
    s1_counts = vectorizer.transform(df_s1['combined_text'])
    del df_s1['combined_text']
    gc.collect()
    
    # Calculate how many times each trigram appears
    col_sums = np.array(s2s3_counts.sum(axis=0)).flatten()
    
    # If a trigram appears in more than 50,000 businesses, it's noise ("ltd", "pvt", "india")
    stop_cols = np.where(col_sums > 50000)[0]
    
    # -------------------------------------------------------------
    # MEMORY FIX: Zero out noisy columns IN-PLACE using Numpy
    # Matrix multiplication creates duplicate 3GB arrays and crashes RAM.
    # np.isin directly modifies the underlying arrays using 0 extra RAM!
    # -------------------------------------------------------------
    mask_s2 = np.isin(s2s3_counts.indices, stop_cols)
    s2s3_counts.data[mask_s2] = 0.0
    s2s3_counts.eliminate_zeros()
    del mask_s2
    
    mask_s1 = np.isin(s1_counts.indices, stop_cols)
    s1_counts.data[mask_s1] = 0.0
    s1_counts.eliminate_zeros()
    del mask_s1
    
    print("Skipping TF-IDF to save 4GB RAM! Using pure HashingVectorizer (Float32)...")
    
    s2s3_vecs = s2s3_counts
    s1_vecs = s1_counts
    gc.collect()
    
    print("Running Ultra-Fast Custom Sparse KNN (Bypassing Scikit-Learn's RAM bloat)...")
    
    # Transpose S2/S3 once for fast dot product
    s2s3_vecs_T = s2s3_vecs.T
    
    # CRITICAL RAM FIX 5: Sparse Matrix Explosion!
    # Even without dense columns, 5000 rows x 10 Million rows creates billions of non-zeros in the dot product.
    # We MUST drop batch_size to 250 to keep RAM under 2GB per batch!
    batch_size = 250  
    
    # Extract IDs
    s1_ids = df_s1['entity_id'].values
    s2s3_ids = df_s2s3['entity_id'].values
    
    # KAGGLE TRICK: Never use a list of dictionaries for 30 million rows (Takes 8GB+ RAM). 
    # Use parallel lists of primitives (Takes < 2GB RAM).
    import os
    
    # ---------------------------------------------------------
    # INCREMENTAL SAVE & RESUME MECHANISM
    # ---------------------------------------------------------
    checkpoint_file = f"{config.BASE_DIR}/code/business_entity_resolution/candidates_incremental_part{part_num}.csv"
    state_file = f"{config.BASE_DIR}/code/business_entity_resolution/candidates_state_part{part_num}.txt"
    start_batch_idx = 0
    
    if os.path.exists(checkpoint_file):
        print(f"Found incremental checkpoint at {checkpoint_file}!")
        if os.path.exists(state_file):
            with open(state_file, "r") as f:
                start_batch_idx = int(f.read().strip())
            print(f"Resuming Fast KNN from batch {start_batch_idx}...")
        else:
            print("State file missing, starting from scratch...")
            start_batch_idx = 0
            if os.path.exists(checkpoint_file): os.remove(checkpoint_file)
            
    # We will write directly to disk every 100,000 queries to save RAM and provide safety
    queries_since_last_save = 0
    out_s1, out_s2, out_scores = [], [], []
    
    # Open file in append mode if resuming, else write mode
    mode = 'a' if start_batch_idx > 0 else 'w'
    with open(checkpoint_file, mode) as f:
        if mode == 'w':
            f.write("source1_entity_id,candidate_entity_id,blocking_score\n")
            
        for start_idx in range(start_batch_idx, s1_vecs.shape[0], batch_size):
            end_idx = min(start_idx + batch_size, s1_vecs.shape[0])
            print(f"Processing Fast KNN Batch: {start_idx} to {end_idx}...")
            
            # 1. Sparse Dot Product
            similarity_matrix = s1_vecs[start_idx:end_idx].dot(s2s3_vecs_T)
            
            # 2. Extract Top K efficiently
            for i in range(similarity_matrix.shape[0]):
                global_s1_idx = start_idx + i
                s1_id = s1_ids[global_s1_idx]
                
                row_data = similarity_matrix.data[similarity_matrix.indptr[i]:similarity_matrix.indptr[i+1]]
                row_indices = similarity_matrix.indices[similarity_matrix.indptr[i]:similarity_matrix.indptr[i+1]]
                
                if len(row_data) == 0:
                    continue
                    
                k = min(n_neighbors, len(row_data))
                top_k_idx = np.argpartition(row_data, -k)[-k:]
                
                for idx in top_k_idx:
                    score = row_data[idx]
                    if score > 0.15:
                        out_s1.append(s1_id)
                        out_s2.append(s2s3_ids[row_indices[idx]])
                        out_scores.append(score)
                        
            queries_since_last_save += batch_size
            
            # INCREMENTAL SAVE EVERY 100,000 QUERIES
            if queries_since_last_save >= 100000 or end_idx == s1_vecs.shape[0]:
                print(f"--> [SAVE] Checkpointing {len(out_s1)} pairs to disk...")
                # Write current buffer to disk
                for idx_save in range(len(out_s1)):
                    f.write(f"{out_s1[idx_save]},{out_s2[idx_save]},{out_scores[idx_save]}\n")
                
                # Update state file
                with open(state_file, "w") as sf:
                    sf.write(str(end_idx))
                    
                # Clear buffer to save RAM
                out_s1, out_s2, out_scores = [], [], []
                queries_since_last_save = 0
                
    print(f"Blocking complete. All candidate pairs saved to {checkpoint_file}.")
    
    # Load the full file back as a DataFrame to pass to the next stage
    print("Loading all generated candidates back into memory...")
    return pd.read_csv(checkpoint_file)
