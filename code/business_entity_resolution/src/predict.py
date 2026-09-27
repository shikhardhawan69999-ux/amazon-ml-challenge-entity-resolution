import pandas as pd
import xgboost as xgb
import os
import gc
from src import config, preprocess, blocking, features

def run_predictions():
    print("--- 1. LOADING TEST SET ---")
    s1 = pd.read_csv(f"{config.TEST_DIR}/test_source1.tsv", sep="\t")
    s2 = pd.read_csv(f"{config.TEST_DIR}/test_source2.tsv", sep="\t")
    s3 = pd.read_csv(f"{config.TEST_DIR}/test_source3.tsv", sep="\t")
    
    s2s3 = pd.concat([s2, s3], ignore_index=True)
    
    del s2, s3
    gc.collect()
    
    print("\n--- 2. PREPROCESSING ---")
    s1 = preprocess.preprocess_dataframe(s1)
    s2s3 = preprocess.preprocess_dataframe(s2s3)
    
    print("\n--- 3. BLOCKING (Ultra-Fast) ---")
    # Generating candidates on test set. Test set might be smaller than train.
    candidates = blocking.generate_candidate_pairs(s1, s2s3, part_num="test")
    
    print("\n--- 4. FEATURE EXTRACTION ---")
    feat_df = features.generate_features(candidates, s1, s2s3)
    
    print("\n--- 5. PREDICTING ON GPU ---")
    feature_cols = ['name_jaro', 'name_ratio', 'name_token_sort', 'name_core_jaro', 'addr_jaro', 'addr_ratio', 'pincode_match', 'country_match', 'blocking_score', 'semantic_similarity']
    
    model = xgb.XGBClassifier(**config.XGB_PARAMS)
    model.load_model(f"{config.BASE_DIR}/code/business_entity_resolution/xgb_model.json")
    
    feat_df['match_prob'] = model.predict_proba(feat_df[feature_cols])[:, 1]
    
    # Keep only high confidence matches
    matches = feat_df[feat_df['match_prob'] > config.MATCH_THRESHOLD].copy()
    
    print("\n--- 6. GENERATING SUBMISSION FILES ---")
    # Group by S1 ID and join matching IDs with comma
    match_grouped = matches.groupby('source1_entity_id')['candidate_entity_id'].apply(lambda x: ','.join(x)).reset_index()
    match_grouped.rename(columns={'candidate_entity_id': 'matched_entity_ids'}, inplace=True)
    
    # Ensure ALL S1 IDs are in the output (even if they have no matches)
    output_df = pd.DataFrame({'source1_entity_id': s1['entity_id']})
    output_df = output_df.merge(match_grouped, on='source1_entity_id', how='left')
    output_df['matched_entity_ids'] = output_df['matched_entity_ids'].fillna("")
    
    out_file = f"{config.OUTPUT_DIR}/matching_results.tsv"
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    output_df.to_csv(out_file, sep='\t', index=False)
    
    print(f"\n🚀 SUCCESS! Final submission file saved to: {out_file}")

if __name__ == "__main__":
    run_predictions()
