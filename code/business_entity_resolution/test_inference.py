import pandas as pd
import xgboost as xgb
from src import config, preprocess, blocking, features

def run_test_inference():
    print("--- 1. LOADING TINY TEST SAMPLE ---")
    s1 = pd.read_csv(f"{config.TEST_DIR}/test_source1.tsv", sep="\t", nrows=50)
    s2 = pd.read_csv(f"{config.TEST_DIR}/test_source2.tsv", sep="\t", nrows=2000)
    s3 = pd.read_csv(f"{config.TEST_DIR}/test_source3.tsv", sep="\t", nrows=2000)
    
    s2s3 = pd.concat([s2, s3], ignore_index=True)
    
    print("\n--- 2. PREPROCESSING ---")
    s1 = preprocess.preprocess_dataframe(s1)
    s2s3 = preprocess.preprocess_dataframe(s2s3)
    
    print("\n--- 3. BLOCKING ---")
    candidates = blocking.generate_candidate_pairs(s1, s2s3, n_neighbors=5)
    
    print("\n--- 4. FEATURE EXTRACTION ---")
    # Patch device to cpu for this environment test
    config.EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
    import src.features as feat_module
    with open('src/features.py', 'r') as f:
        code = f.read().replace("device='cuda'", "device='cpu'")
    with open('src/features_temp.py', 'w') as f:
        f.write(code)
    import src.features_temp as temp_feat
    
    feat_df = temp_feat.generate_features(candidates, s1, s2s3)
    
    print("\n--- 5. PREDICTING ---")
    feature_cols = ['name_jaro', 'name_ratio', 'name_token_sort', 'name_core_jaro', 'addr_jaro', 'addr_ratio', 'pincode_match', 'country_match', 'blocking_score', 'semantic_similarity']
    
    model = xgb.XGBClassifier()
    model.load_model("xgb_model.json")
    
    feat_df['match_prob'] = model.predict_proba(feat_df[feature_cols])[:, 1]
    
    # Filter matches
    matches = feat_df[feat_df['match_prob'] > 0.5] # Lower threshold since model was trained on tiny dummy data
    
    print("\n===============================")
    print("      SAMPLE PREDICTIONS       ")
    print("===============================\n")
    
    s1_dict = s1.set_index('entity_id').to_dict('index')
    s2s3_dict = s2s3.set_index('entity_id').to_dict('index')
    
    count = 0
    for idx, row in matches.sort_values(by='match_prob', ascending=False).iterrows():
        if count >= 10: break
        source = s1_dict[row['source1_entity_id']]
        candidate = s2s3_dict[row['candidate_entity_id']]
        
        print(f"Match Confidence: {row['match_prob']*100:.1f}%")
        print(f"[TARGET S1] {source['business_name']} | {source['business_address']}")
        print(f"[FOUND S2/3] {candidate['business_name']} | {candidate['business_address']}")
        print("-" * 50)
        count += 1
        
    print("\nTest inference successfully completed!")

if __name__ == "__main__":
    run_test_inference()
