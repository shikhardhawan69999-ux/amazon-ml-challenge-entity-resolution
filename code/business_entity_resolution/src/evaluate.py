import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from src import config, preprocess, blocking, features

def calculate_macro_f05(ground_truth_dict, predictions_dict, all_s1_ids):
    """
    Calculates the exact Macro-Averaged F0.5 score as described in the challenge.
    """
    f05_scores = []
    
    for s1_id in all_s1_ids:
        gt_matches = set(ground_truth_dict.get(s1_id, []))
        pred_matches = set(predictions_dict.get(s1_id, []))
        
        # Singleton logic
        if len(gt_matches) == 0:
            if len(pred_matches) == 0:
                f05_scores.append(1.0) # Correctly identified as singleton
            else:
                f05_scores.append(0.0) # Falsely assigned matches to a singleton
            continue
            
        tp = len(gt_matches.intersection(pred_matches))
        fp = len(pred_matches - gt_matches)
        fn = len(gt_matches - pred_matches)
        
        if tp == 0:
            f05_scores.append(0.0)
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f05 = (1.25 * precision * recall) / ((0.25 * precision) + recall)
            f05_scores.append(f05)
            
    # Macro-average
    return sum(f05_scores) / len(f05_scores)

def run_evaluation():
    print("Loading training data for evaluation split...")
    s1 = pd.read_csv(f"{config.TRAIN_DIR}/train_source1.tsv", sep="\t")
    s2 = pd.read_csv(f"{config.TRAIN_DIR}/train_source2.tsv", sep="\t")
    s3 = pd.read_csv(f"{config.TRAIN_DIR}/train_source3.tsv", sep="\t")
    gt = pd.read_csv(f"{config.TRAIN_DIR}/train_ground_truth.tsv", sep="\t").fillna("")
    
    # Split S1 into 80% train, 20% validation
    s1_train, s1_val = train_test_split(s1, test_size=0.2, random_state=42)
    print(f"Split data: {len(s1_train)} Train entities, {len(s1_val)} Validation entities")
    
    s2s3 = pd.concat([s2, s3], ignore_index=True)
    
    print("\n--- PREPROCESSING ---")
    s1_val = preprocess.preprocess_dataframe(s1_val)
    s2s3 = preprocess.preprocess_dataframe(s2s3)
    
    print("\n--- BLOCKING (Candidate Generation) ---")
    candidates = blocking.generate_candidate_pairs(s1_val, s2s3)
    
    print("\n--- FEATURE ENGINEERING ---")
    feat_df = features.generate_features(candidates, s1_val, s2s3)
    feature_cols = ['name_jaro', 'name_ratio', 'name_token_sort', 'name_core_jaro', 'addr_jaro', 'addr_ratio', 'pincode_match', 'country_match', 'blocking_score']
    
    print("\n--- PREDICTING ---")
    model = xgb.XGBClassifier()
    model.load_model(f"{config.BASE_DIR}/code/business_entity_resolution/xgb_model.json")
    
    feat_df['match_prob'] = model.predict_proba(feat_df[feature_cols])[:, 1]
    
    # Test different thresholds to find the best F0.5
    gt_dict = dict(zip(gt['source1_entity_id'], gt['matched_entity_ids'].apply(lambda x: x.split(',') if x else [])))
    all_s1_ids = s1_val['entity_id'].tolist()
    
    print("\n--- RESULTS SUMMARY ---")
    for threshold in [0.50, 0.70, 0.85, 0.88, 0.90, 0.95]:
        matches = feat_df[feat_df['match_prob'] > threshold]
        pred_grouped = matches.groupby('source1_entity_id')['candidate_entity_id'].apply(list).to_dict()
        
        score = calculate_macro_f05(gt_dict, pred_grouped, all_s1_ids)
        print(f"Threshold: {threshold:.2f} --> F0.5 Score: {score:.5f}")
        
    print("\nDetailed Evaluation Complete! Set the highest performing threshold in config.py!")

if __name__ == "__main__":
    run_evaluation()
