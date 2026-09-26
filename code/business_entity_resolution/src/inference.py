import pandas as pd
import xgboost as xgb
import os
from src import config, preprocess, blocking, features

def run_inference():
    print("Loading test data...")
    s1 = pd.read_csv(f"{config.TEST_DIR}/test_source1.tsv", sep="\t")
    s2 = pd.read_csv(f"{config.TEST_DIR}/test_source2.tsv", sep="\t")
    s3 = pd.read_csv(f"{config.TEST_DIR}/test_source3.tsv", sep="\t")
    
    s2s3 = pd.concat([s2, s3], ignore_index=True)
    
    print("Preprocessing...")
    s1 = preprocess.preprocess_dataframe(s1)
    s2s3 = preprocess.preprocess_dataframe(s2s3)
    
    candidates = blocking.generate_candidate_pairs(s1, s2s3)
    
    print("Exporting candidate pairs...")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # Save candidate_pairs.tsv correctly formatted
    candidates_grouped = candidates.groupby('source1_entity_id')['candidate_entity_id'].apply(lambda x: ','.join(x)).reset_index()
    candidate_out = s1[['entity_id']].rename(columns={'entity_id': 'source1_entity_id'})
    candidate_out = candidate_out.merge(candidates_grouped, on='source1_entity_id', how='left').fillna('')
    candidate_out.rename(columns={'candidate_entity_id': 'candidate_entity_ids'}, inplace=True)
    
    candidate_out.to_csv(f"{config.OUTPUT_DIR}/candidate_pairs.tsv", sep="\t", index=False)
    
    feat_df = features.generate_features(candidates, s1, s2s3)
    feature_cols = ['name_jaro', 'name_ratio', 'name_token_sort', 'name_core_jaro', 'addr_jaro', 'addr_ratio', 'pincode_match', 'country_match', 'blocking_score', 'semantic_similarity']
    
    print("Running Inference...")
    model = xgb.XGBClassifier()
    model.load_model(f"{config.BASE_DIR}/code/business_entity_resolution/xgb_model.json")
    
    feat_df['match_prob'] = model.predict_proba(feat_df[feature_cols])[:, 1]
    
    # Apply strict F0.5 precision threshold
    print(f"Applying High-Precision Threshold: {config.MATCH_THRESHOLD}")
    matches = feat_df[feat_df['match_prob'] > config.MATCH_THRESHOLD]
    
    # ---------------------------------------------------------
    # HACKATHON MASTERSTROKE: Bipartite 1-to-N Constraint
    # A noisy record in S2/S3 can only belong to ONE real-world entity in S1.
    # If the model assigns a noisy record to two different clean businesses, 
    # we ONLY keep the one with the highest probability.
    # This drops thousands of False Positives and skyrockets the F0.5 Precision!
    # ---------------------------------------------------------
    print("Enforcing strictly disjoint S1 entities to maximize Precision...")
    matches = matches.sort_values('match_prob', ascending=False)
    matches = matches.drop_duplicates(subset=['candidate_entity_id'], keep='first')
    
    matches_grouped = matches.groupby('source1_entity_id')['candidate_entity_id'].apply(lambda x: ','.join(x)).reset_index()
    matches_grouped.rename(columns={'candidate_entity_id': 'matched_entity_ids'}, inplace=True)
    
    final_output = s1[['entity_id']].rename(columns={'entity_id': 'source1_entity_id'})
    final_output = final_output.merge(matches_grouped, on='source1_entity_id', how='left').fillna('')
    
    final_output.to_csv(f"{config.OUTPUT_DIR}/matching_results.tsv", sep="\t", index=False)
    print("Inference complete. Output saved to matching_results.tsv")

if __name__ == "__main__":
    run_inference()
