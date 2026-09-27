import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import glob
import os
from src import config

def run_merge_and_train():
    print("Looking for feature files (features_partX.csv)...")
    
    # Find all generated feature parts in the directory
    feature_files = glob.glob(f"{config.BASE_DIR}/code/business_entity_resolution/features_part*.csv")
    
    if not feature_files:
        print("Error: No features_partX.csv files found! Run train.py first.")
        return
        
    print(f"Found {len(feature_files)} parts! Merging them together...")
    
    # Read and merge all parts
    df_list = [pd.read_csv(f) for f in feature_files]
    df = pd.concat(df_list, ignore_index=True)
    
    print(f"Successfully merged! Total Training Rows: {len(df)}")
    
    feature_cols = ['name_jaro', 'name_ratio', 'name_token_sort', 'name_core_jaro', 
                    'addr_jaro', 'addr_ratio', 'pincode_match', 'country_match', 
                    'blocking_score', 'semantic_similarity']
                    
    X = df[feature_cols]
    y = df['label']
    
    print("\n[1/2] Training XGBoost Classifier on RTX 4050...")
    xgb_model = xgb.XGBClassifier(**config.XGB_PARAMS)
    xgb_model.fit(X, y)
    
    xgb_model_path = f"{config.BASE_DIR}/code/business_entity_resolution/xgb_model.json"
    xgb_model.save_model(xgb_model_path)
    print(f"XGBoost Model saved to {xgb_model_path}")
    
    print("\n[2/2] Training LightGBM Classifier on RTX 4050...")
    lgbm_model = lgb.LGBMClassifier(**config.LGBM_PARAMS)
    lgbm_model.fit(X, y)
    
    lgbm_model_path = f"{config.BASE_DIR}/code/business_entity_resolution/lgbm_model.txt"
    lgbm_model.booster_.save_model(lgbm_model_path)
    print(f"LightGBM Model saved to {lgbm_model_path}")
    
    print("\nDONE! Models are ready to predict on the Test Set!")

if __name__ == "__main__":
    run_merge_and_train()
