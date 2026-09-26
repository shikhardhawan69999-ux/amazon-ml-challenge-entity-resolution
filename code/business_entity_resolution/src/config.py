import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.path.join(BASE_DIR, "student_resource", "dataset")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Blocking parameters (Tuning recall vs runtime)
TFIDF_NGRAM_RANGE = (3, 3) # Restrict to trigrams only to prevent massive RAM spikes
TFIDF_MAX_FEATURES = 30000  # Lowered drastically to prevent 24GB RAM OOM
KNN_NEIGHBORS = 15          # Lowered to 15 to speed up processing of 55 million pairs

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
    'n_jobs': -1, # <--- FORCE ALL 16 RYZEN THREADS
    'n_estimators': 800,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}

# Post-processing threshold optimized for F0.5 (Precision-heavy)
MATCH_THRESHOLD = 0.88
