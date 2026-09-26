import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.neighbors import NearestNeighbors
import gc
from src import config

import numpy as np

def generate_candidate_pairs(df_s1, df_s2s3, n_neighbors=config.KNN_NEIGHBORS):
    print("Vectorizing text for blocking (Using HashingVectorizer for Ultra-Low RAM)...")
    
    # HashingVectorizer requires ZERO RAM for vocabulary building
    # dtype=np.float32 cuts the final sparse matrix memory strictly in HALF
    vectorizer = HashingVectorizer(analyzer='char_wb', 
                                   ngram_range=config.TFIDF_NGRAM_RANGE, 
                                   n_features=config.TFIDF_MAX_FEATURES,
                                   norm=None, 
                                   alternate_sign=False,
                                   dtype=np.float32)
    
    print("Applying TF-IDF Weights (In-Place to save 4GB RAM)...")
    # copy=False prevents TfidfTransformer from creating a duplicate 3GB sparse matrix
    tfidf = TfidfTransformer(copy=False) 
    
    s2s3_counts = vectorizer.transform(df_s2s3['combined_text'])
    tfidf.fit(s2s3_counts) # Fit weights
    s2s3_vecs = tfidf.transform(s2s3_counts) # Transforms IN-PLACE
    del s2s3_counts
    gc.collect()
    
    s1_counts = vectorizer.transform(df_s1['combined_text'])
    s1_vecs = tfidf.transform(s1_counts) # Transforms IN-PLACE
    del s1_counts
    gc.collect()
    
    print("Running Ultra-Fast Custom Sparse KNN (Bypassing Scikit-Learn's RAM bloat)...")
    
    # Transpose S2/S3 once for fast dot product
    s2s3_vecs_T = s2s3_vecs.T
    
    # We can now safely use a much larger batch size because the output remains SPARSE!
    batch_size = 5000  
    
    # Extract IDs
    s1_ids = df_s1['entity_id'].values
    s2s3_ids = df_s2s3['entity_id'].values
    
    # KAGGLE TRICK: Never use a list of dictionaries for 30 million rows (Takes 8GB+ RAM). 
    # Use parallel lists of primitives (Takes < 2GB RAM).
    out_s1 = []
    out_s2 = []
    out_scores = []
    
    import numpy as np
    
    for start_idx in range(0, s1_vecs.shape[0], batch_size):
        end_idx = min(start_idx + batch_size, s1_vecs.shape[0])
        print(f"Processing Fast KNN Batch: {start_idx} to {end_idx}...")
        
        # 1. Sparse Dot Product (Result is SPARSE, taking only MBs instead of GBs of RAM)
        similarity_matrix = s1_vecs[start_idx:end_idx].dot(s2s3_vecs_T)
        
        # 2. Extract Top K efficiently
        for i in range(similarity_matrix.shape[0]):
            global_s1_idx = start_idx + i
            s1_id = s1_ids[global_s1_idx]
            
            # Get only the non-zero similarities for this specific query
            row_data = similarity_matrix.data[similarity_matrix.indptr[i]:similarity_matrix.indptr[i+1]]
            row_indices = similarity_matrix.indices[similarity_matrix.indptr[i]:similarity_matrix.indptr[i+1]]
            
            if len(row_data) == 0:
                continue
                
            # Find the indices of the top K elements
            k = min(n_neighbors, len(row_data))
            
            # np.argpartition is extremely fast for finding top K
            top_k_idx = np.argpartition(row_data, -k)[-k:]
            
            for idx in top_k_idx:
                score = row_data[idx]
                if score > 0.15:
                    out_s1.append(s1_id)
                    out_s2.append(s2s3_ids[row_indices[idx]])
                    out_scores.append(score)
                    
    print(f"Blocking complete. Generated {len(out_s1)} candidate pairs.")
    return pd.DataFrame({
        'source1_entity_id': out_s1,
        'candidate_entity_id': out_s2,
        'blocking_score': out_scores
    })
