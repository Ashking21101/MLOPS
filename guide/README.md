# ML Project with MLflow & AWS SageMaker

Simple ML project with MLflow experiment tracking and optional SageMaker integration.

## Quick Start

```bash
# Setup
pip install -r requirements.txt

# Generate sample data (optional)
python generate_sample_data.py

# Train locally
python train_local.py

# View results
mlflow ui  # http://localhost:5000
```

## Structure

```
config/           → model & training config
src/              → data, model, eval, inference code
data/             → raw & processed data
models/           → saved model artifacts
sagemaker/        → SageMaker training script
train_local.py    → local training entry point
train_sagemaker.py → SageMaker job submission
```

## Usage

### Local Training
```bash
python train_local.py
```

### SageMaker Training
```bash
# Set AWS credentials first
aws configure

# Submit job
python train_sagemaker.py --instance-type ml.m5.large
```

### Make Predictions
```python
from src.inference import load_model, load_scaler, predict
import pandas as pd

model = load_model("models/model.pkl")
scaler = load_scaler("models/scaler.pkl")

data = pd.read_csv("new_data.csv")
predictions = predict(model, scaler, data)
```

## Configuration

Edit `config/config.yaml`:
- Model parameters (n_estimators, max_depth, etc.)
- Data paths
- AWS settings

## Next Steps

- Add hyperparameter tuning
- Add more model types
- Set up CI/CD pipeline
- Deploy to SageMaker Endpoint

