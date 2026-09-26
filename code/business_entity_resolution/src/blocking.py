import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from src import config

def generate_candidate_pairs(df_s1, df_s2s3, n_neighbors=config.KNN_NEIGHBORS):
    print("Vectorizing text for blocking...")
    vectorizer = TfidfVectorizer(analyzer='char_wb', 
                                 ngram_range=config.TFIDF_NGRAM_RANGE, 
                                 max_features=config.TFIDF_MAX_FEATURES)
    
    # Fit on all available data to create a robust shared vocabulary
    all_text = pd.concat([df_s1['combined_text'], df_s2s3['combined_text']])
    vectorizer.fit(all_text)
    
    s1_vecs = vectorizer.transform(df_s1['combined_text'])
    s2s3_vecs = vectorizer.transform(df_s2s3['combined_text'])
    
    print("Running Nearest Neighbors for candidate generation...")
    # Cosine distance naturally handles varying lengths of entity names/addresses
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', n_jobs=-1)
    nn.fit(s2s3_vecs)
    
    distances, indices = nn.kneighbors(s1_vecs)
    
    pairs = []
    # Create pairs dataframe
    for i, s1_idx in enumerate(df_s1.index):
        s1_id = df_s1.loc[s1_idx, 'entity_id']
        for j in range(n_neighbors):
            s2s3_idx = df_s2s3.index[indices[i, j]]
            s2s3_id = df_s2s3.loc[s2s3_idx, 'entity_id']
            score = 1.0 - distances[i, j] # Convert cosine distance to cosine similarity
            
            if score > 0.15: # Cutoff to drop absolute noise and keep candidate list compact
                pairs.append({'source1_entity_id': s1_id, 'candidate_entity_id': s2s3_id, 'blocking_score': score})
                
    return pd.DataFrame(pairs)
