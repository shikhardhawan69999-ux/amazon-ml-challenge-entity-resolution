import pandas as pd
import xgboost as xgb
import os
from src import config, preprocess, blocking, features

def run_end_to_end_test():
    print("--- 1. LOADING TINY SAMPLE ---")
    s1 = pd.read_csv(f"{config.TRAIN_DIR}/train_source1.tsv", sep="\t", nrows=500)
    s2 = pd.read_csv(f"{config.TRAIN_DIR}/train_source2.tsv", sep="\t", nrows=2000)
    s3 = pd.read_csv(f"{config.TRAIN_DIR}/train_source3.tsv", sep="\t", nrows=2000)
    gt = pd.read_csv(f"{config.TRAIN_DIR}/train_ground_truth.tsv", sep="\t")
    
    s2s3 = pd.concat([s2, s3], ignore_index=True)
    
    print("\n--- 2. PREPROCESSING ---")
    s1 = preprocess.preprocess_dataframe(s1)
    s2s3 = preprocess.preprocess_dataframe(s2s3)
    
    print("\n--- 3. BLOCKING (TF-IDF + KNN) ---")
    candidates = blocking.generate_candidate_pairs(s1, s2s3, n_neighbors=5)
    
    print("\n--- 4. FEATURE EXTRACTION ---")
    # Patch features.py to use CPU for this test
    config.EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
    import src.features as feat_module
    with open('src/features.py', 'r') as f:
        code = f.read().replace("device='cuda'", "device='cpu'")
    with open('src/features_temp.py', 'w') as f:
        f.write(code)
    import src.features_temp as temp_feat
    
    feat_df = temp_feat.generate_features(candidates, s1, s2s3)
    
    print("\n--- 5. GENERATING LABELS ---")
    gt['matched_entity_ids'] = gt['matched_entity_ids'].fillna("")
    gt_dict = dict(zip(gt['source1_entity_id'], gt['matched_entity_ids'].str.split(',')))
    
    def check_match(row):
        matches = gt_dict.get(row['source1_entity_id'], [])
        return 1 if row['candidate_entity_id'] in matches else 0
        
    feat_df['label'] = feat_df.apply(check_match, axis=1)
    
    print("\n--- 6. TRAINING XGBOOST ---")
    feature_cols = ['name_jaro', 'name_ratio', 'name_token_sort', 'name_core_jaro', 'addr_jaro', 'addr_ratio', 'pincode_match', 'country_match', 'blocking_score', 'semantic_similarity']
    
    X = feat_df[feature_cols]
    y = feat_df['label']
    
    # We use cpu predictor for the test
    config.XGB_PARAMS['device'] = 'cpu'
    model = xgb.XGBClassifier(**config.XGB_PARAMS)
    model.fit(X, y)
    model.save_model("xgb_model.json")
    
    # Clean up temp file
    if os.path.exists('src/features_temp.py'):
        os.remove('src/features_temp.py')
    
    print("\nEnd-to-End Test Passed! The algorithm successfully extracted features and trained the model.")

if __name__ == "__main__":
    run_end_to_end_test()
