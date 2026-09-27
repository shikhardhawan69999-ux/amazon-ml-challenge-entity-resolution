import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import os
import numpy as np
from src import config, inference

def test_inf():
    print("Patching config for fast test...")
    # Mock the test files using nrows
    orig_read_csv = pd.read_csv
    def mock_read_csv(*args, **kwargs):
        kwargs['nrows'] = 500
        return orig_read_csv(*args, **kwargs)
    pd.read_csv = mock_read_csv
    
    # Mock models
    print("Generating dummy models...")
    X_dummy = pd.DataFrame(0, index=np.arange(10), columns=['name_jaro', 'name_ratio', 'name_token_sort', 'name_core_jaro', 'addr_jaro', 'addr_ratio', 'pincode_match', 'country_match', 'blocking_score', 'semantic_similarity'])
    y_dummy = pd.Series([0,1]*5)
    
    xgb_m = xgb.XGBClassifier()
    xgb_m.fit(X_dummy, y_dummy)
    xgb_m.save_model("xgb_model.json")
    
    lgbm_m = lgb.LGBMClassifier()
    lgbm_m.fit(X_dummy, y_dummy)
    lgbm_m.booster_.save_model("lgbm_model.txt")
    
    # Mock CPU features
    config.EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
    with open('src/features.py', 'r') as f:
        code = f.read().replace("device='cuda'", "device='cpu'")
    with open('src/features_temp2.py', 'w') as f:
        f.write(code)
    
    import src.features_temp2 as temp_feat
    inference.features = temp_feat
    
    print("Running Inference...")
    try:
        inference.run_inference()
        print("INFERENCE TEST PASSED!")
    finally:
        pd.read_csv = orig_read_csv
        if os.path.exists('src/features_temp2.py'): os.remove('src/features_temp2.py')

if __name__ == '__main__':
    test_inf()
