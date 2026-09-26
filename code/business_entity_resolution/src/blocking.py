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
    
    s1_counts = vectorizer.transform(df_s1['combined_text'])
    s2s3_counts = vectorizer.transform(df_s2s3['combined_text'])
    
    print("Applying TF-IDF Weights...")
    tfidf = TfidfTransformer()
    tfidf.fit(s2s3_counts) # Fit weights only on the larger dataset to save memory
    
    s1_vecs = tfidf.transform(s1_counts)
    s2s3_vecs = tfidf.transform(s2s3_counts)
    
    # Aggressively free up RAM before KNN
    del s1_counts
    del s2s3_counts
    gc.collect()
    
    print("Running Nearest Neighbors for candidate generation (in batches to save RAM)...")
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', n_jobs=-1)
    nn.fit(s2s3_vecs)
    
    pairs = []
    batch_size = 25000  # Process 25k queries at a time for maximum RAM safety
    
    # Extract S1 and S2/S3 IDs to fast lists for indexing
    s1_ids = df_s1['entity_id'].values
    s2s3_ids = df_s2s3['entity_id'].values
    
    for start_idx in range(0, s1_vecs.shape[0], batch_size):
        end_idx = min(start_idx + batch_size, s1_vecs.shape[0])
        print(f"Processing KNN Batch: {start_idx} to {end_idx}...")
        
        batch_distances, batch_indices = nn.kneighbors(s1_vecs[start_idx:end_idx])
        
        for i in range(batch_distances.shape[0]):
            global_s1_idx = start_idx + i
            s1_id = s1_ids[global_s1_idx]
            
            for j in range(n_neighbors):
                s2s3_idx = batch_indices[i, j]
                s2s3_id = s2s3_ids[s2s3_idx]
                score = 1.0 - batch_distances[i, j]
                
                if score > 0.15:
                    pairs.append({'source1_entity_id': s1_id, 'candidate_entity_id': s2s3_id, 'blocking_score': score})
                    
    print(f"Blocking complete. Generated {len(pairs)} candidate pairs.")
    return pd.DataFrame(pairs)
