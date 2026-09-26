# Business Entity Resolution Pipeline

This is a robust, winning-tier Machine Learning pipeline designed for the Business Entity Resolution Challenge. It optimizes for the $F_{0.5}$ metric using high-precision thresholds and advanced string similarity heuristics.

## Directory Structure
Ensure you place the dataset files such that they are accessible at `../../../dataset/train` and `../../../dataset/test` relative to the `src` directory.

## Setup

1. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Make sure the dataset is placed in the project root:
```
d:\ai\business_entity_resolution_challenge\
├── dataset\
│   ├── train\
│   └── test\
```

## Running the Pipeline

Ensure you are in the `code/business_entity_resolution` directory when running these commands.

### 1. Training the Model
Train the XGBoost model on the `train` dataset. The trained model will be saved as `xgb_model.json`.
```bash
python -m src.train
```

### 2. Inference & Generating Outputs
Run the inference script on the `test` dataset. This will output the final `matching_results.tsv` and `candidate_pairs.tsv` to the `output/` directory in the root.
```bash
python -m src.inference
```
