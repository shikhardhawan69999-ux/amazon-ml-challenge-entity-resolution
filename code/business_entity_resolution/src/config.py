import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Blocking parameters (Tuning recall vs runtime)
TFIDF_NGRAM_RANGE = (2, 4)
TFIDF_MAX_FEATURES = 150000
KNN_NEIGHBORS = 25

# Feature engineering parameters
EMBEDDING_MODEL = 'all-MiniLM-L6-v2' # 22M param model, extremely fast, MIT licensed

# Model parameters (XGBoost)
XGB_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'aucpr', 
    'learning_rate': 0.05,
    'max_depth': 6,
    'tree_method': 'hist',
    'device': 'cuda', # <--- ENABLE GPU ACCELERATION
    'n_estimators': 800,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}

# Post-processing threshold optimized for F0.5 (Precision-heavy)
MATCH_THRESHOLD = 0.88
