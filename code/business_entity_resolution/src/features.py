import pandas as pd
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import paired_cosine_distances
import numpy as np
from src import config

def generate_features(candidate_pairs, df_s1, df_s2s3):
    print("Generating string distance features (Optimized for large data)...")
    
    # Convert dataframes to fast dictionary lookups to avoid slow Pandas indexing during loops
    s1_dict = df_s1.set_index('entity_id').to_dict('index')
    s2s3_dict = df_s2s3.set_index('entity_id').to_dict('index')
    
    # Prepare lists for fast appending
    name_jaro, name_ratio, name_token_sort = [], [], []
    name_core_jaro, pincode_match = [], []
    addr_jaro, addr_ratio, country_match = [], [], []
    s1_texts, s2s3_texts = [], []
    
    # Iterate over tuples (very fast compared to iterrows or apply)
    from rapidfuzz.distance import JaroWinkler
    for row in candidate_pairs.itertuples(index=False):
        s1 = s1_dict[row.source1_entity_id]
        s2 = s2s3_dict[row.candidate_entity_id]
        
        n1, n2 = s1['business_name_clean'], s2['business_name_clean']
        n1_core, n2_core = s1['business_name_core'], s2['business_name_core']
        a1, a2 = s1['business_address_clean'], s2['business_address_clean']
        
        name_jaro.append(JaroWinkler.normalized_similarity(n1, n2))
        name_ratio.append(fuzz.ratio(n1, n2) / 100.0)
        name_token_sort.append(fuzz.token_sort_ratio(n1, n2) / 100.0)
        
        name_core_jaro.append(JaroWinkler.normalized_similarity(n1_core, n2_core))
        
        addr_jaro.append(JaroWinkler.normalized_similarity(a1, a2))
        addr_ratio.append(fuzz.ratio(a1, a2) / 100.0)
        
        # Pincode extraction logic
        p1, p2 = s1['pincode'], s2['pincode']
        if p1 and p2 and p1 == p2:
            pincode_match.append(1)
        elif p1 and p2 and p1 != p2:
            pincode_match.append(-1)
        else:
            pincode_match.append(0)
            
        country_match.append(int(s1['country'] == s2['country']))
        
    features_df = pd.DataFrame({
        'name_jaro': name_jaro,
        'name_ratio': name_ratio,
        'name_token_sort': name_token_sort,
        'name_core_jaro': name_core_jaro,
        'addr_jaro': addr_jaro,
        'addr_ratio': addr_ratio,
        'pincode_match': pincode_match,
        'country_match': country_match
    })
    
    print("Generating semantic embeddings (GPU Accelerated)...")
    model = SentenceTransformer(config.EMBEDDING_MODEL, device='cuda')
    
    # -------------------------------------------------------------
    # GPU OPTIMIZATION: Only encode UNIQUE businesses (12.5M total)
    # Instead of encoding the redundant pairs (33M total).
    # This cuts GPU processing time from 2 hours to 20 minutes!
    # -------------------------------------------------------------
    print("Encoding Source 1 unique businesses...")
    s1_unique_texts = df_s1['combined_text'].fillna("").tolist()
    s1_emb = model.encode(s1_unique_texts, batch_size=256, show_progress_bar=True)
    
    print("Encoding Source 2/3 unique businesses...")
    s2s3_unique_texts = df_s2s3['combined_text'].fillna("").tolist()
    s2s3_emb = model.encode(s2s3_unique_texts, batch_size=256, show_progress_bar=True)
    
    # Map entity_id to their integer index in the arrays
    s1_id_to_idx = {id_: idx for idx, id_ in enumerate(df_s1['entity_id'])}
    s2s3_id_to_idx = {id_: idx for idx, id_ in enumerate(df_s2s3['entity_id'])}
    
    # Pull the exact embeddings for the 33M pairs
    print("Mapping embeddings to candidate pairs...")
    s1_pair_indices = [s1_id_to_idx[id_] for id_ in candidate_pairs['source1_entity_id']]
    s2s3_pair_indices = [s2s3_id_to_idx[id_] for id_ in candidate_pairs['candidate_entity_id']]
    
    emb1 = s1_emb[s1_pair_indices]
    emb2 = s2s3_emb[s2s3_pair_indices]
    
    # Compute cosine similarity between the embeddings
    print("Calculating final Cosine Similarities...")
    cosine_sim = 1 - paired_cosine_distances(emb1, emb2)
    features_df['semantic_similarity'] = cosine_sim
    
    return pd.concat([candidate_pairs, features_df], axis=1)
