import pandas as pd
import xgboost as xgb
import os
from src import config, preprocess, blocking, features

def load_and_prepare_training_data():
    features_ckpt = f"{config.BASE_DIR}/code/business_entity_resolution/features_checkpoint.csv"
    candidates_ckpt = f"{config.BASE_DIR}/code/business_entity_resolution/candidates_checkpoint.csv"
    
    # CHECKPOINT 2: If final features already exist, skip EVERYTHING and load it!
    if os.path.exists(features_ckpt):
        print("\n[CHECKPOINT] Found existing final features! Loading directly... (Skipping all previous heavy steps)")
        return pd.read_csv(features_ckpt)
        
    # If no final features, load the raw data
    print("Loading training data...")
    s1 = pd.read_csv(f"{config.TRAIN_DIR}/train_source1.tsv", sep="\t")
    
    print("\n[DEBUG] Taking a 5,00,000 row sample of S1 to meet the 6 PM Deadline! (2 hour run)")
    s1 = s1.head(500000).copy()
    
    s2 = pd.read_csv(f"{config.TRAIN_DIR}/train_source2.tsv", sep="\t")
    s3 = pd.read_csv(f"{config.TRAIN_DIR}/train_source3.tsv", sep="\t")
    gt = pd.read_csv(f"{config.TRAIN_DIR}/train_ground_truth.tsv", sep="\t")
    s2s3 = pd.concat([s2, s3], ignore_index=True)
    
    print("Preprocessing...")
    s1 = preprocess.preprocess_dataframe(s1)
    s2s3 = preprocess.preprocess_dataframe(s2s3)
    
    s1_clean_path = f"{config.BASE_DIR}/code/business_entity_resolution/s1_clean.pkl"
    s2s3_clean_path = f"{config.BASE_DIR}/code/business_entity_resolution/s2s3_clean.pkl"
    
    import gc
    # CHECKPOINT 1: Check if Blocking is already done
    if os.path.exists(candidates_ckpt):
        print("\n[CHECKPOINT] Found existing Blocking pairs! Loading candidates directly...")
        candidates = pd.read_csv(candidates_ckpt)
        # We still need to save to disk because features step expects it from disk now
        s1.to_pickle(s1_clean_path)
        s2s3.to_pickle(s2s3_clean_path)
    else:
        print("\n[RAM OPTIMIZATION] Offloading 12GB of extra Pandas columns to disk...")
        # Save FULL dataframes to disk
        s1.to_pickle(s1_clean_path)
        s2s3.to_pickle(s2s3_clean_path)
        
        # Keep ONLY the 2 columns needed for blocking to free up maximum RAM
        s1_blocking = s1[['entity_id', 'combined_text']].copy()
        s2s3_blocking = s2s3[['entity_id', 'combined_text']].copy()
        
        # Physically delete the massive 12GB dataframes from RAM
        del s1
        del s2s3
        gc.collect()
        
        candidates = blocking.generate_candidate_pairs(s1_blocking, s2s3_blocking)
        print(f"\n[CHECKPOINT] Saving Blocking pairs to {candidates_ckpt}...")
        candidates.to_csv(candidates_ckpt, index=False)
        
        del s1_blocking
        del s2s3_blocking
        gc.collect()
    
    print("\n[RAM OPTIMIZATION] Reloading full Pandas data from disk for Feature Extraction...")
    s1 = pd.read_pickle(s1_clean_path)
    s2s3 = pd.read_pickle(s2s3_clean_path)
        
    feat_df = features.generate_features(candidates, s1, s2s3)
    
    print("Generating ground truth labels...")
    # Create labels based on ground truth
    gt['matched_entity_ids'] = gt['matched_entity_ids'].fillna("")
    gt_dict = dict(zip(gt['source1_entity_id'], gt['matched_entity_ids'].str.split(',')))
    
    def check_match(row):
        matches = gt_dict.get(row['source1_entity_id'], [])
        return 1 if row['candidate_entity_id'] in matches else 0
        
    feat_df['label'] = feat_df.apply(check_match, axis=1)
    
    # Save final features checkpoint before returning
    print(f"\n[CHECKPOINT] Saving Final Features to {features_ckpt}...")
    feat_df.to_csv(features_ckpt, index=False)
    
    return feat_df

def train_model():
    df = load_and_prepare_training_data()
    feature_cols = ['name_jaro', 'name_ratio', 'name_token_sort', 'name_core_jaro', 'addr_jaro', 'addr_ratio', 'pincode_match', 'country_match', 'blocking_score', 'semantic_similarity']
    
    X = df[feature_cols]
    y = df['label']
    
    print("\n[1/2] Training XGBoost Classifier on RTX 4050...")
    xgb_model = xgb.XGBClassifier(**config.XGB_PARAMS)
    xgb_model.fit(X, y)
    
    xgb_model_path = f"{config.BASE_DIR}/code/business_entity_resolution/xgb_model.json"
    xgb_model.save_model(xgb_model_path)
    print(f"XGBoost Model saved to {xgb_model_path}")
    
    print("\n[2/2] Training LightGBM Classifier on RTX 4050...")
    import lightgbm as lgb
    lgbm_model = lgb.LGBMClassifier(**config.LGBM_PARAMS)
    lgbm_model.fit(X, y)
    
    lgbm_model_path = f"{config.BASE_DIR}/code/business_entity_resolution/lgbm_model.txt"
    lgbm_model.booster_.save_model(lgbm_model_path)
    print(f"LightGBM Model saved to {lgbm_model_path}")
    
    print("\n✅ Ensemble Dual-Training Complete! Models are ready for inference.")

if __name__ == "__main__":
    train_model()
