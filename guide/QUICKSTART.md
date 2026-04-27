# Quick Start Guide

## 1. Installation

```bash
# Clone or navigate to project
cd ml-sagemaker-mlflow

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 2. Prepare Your Data

Place your training CSV file in `data/raw/`:
```
data/raw/data.csv
```

**Required CSV format:**
- Columns: features + target column (default: "target")
- Example:
  ```
  feature1,feature2,feature3,...,target
  1.5,2.3,4.1,...,0
  2.1,3.4,5.2,...,1
  ```

## 3. Update Configuration

Edit `config/config.yaml`:
```yaml
data:
  raw_path: "data/raw/YOUR_DATA.csv"
  target_column: "YOUR_TARGET_COLUMN"

model:
  n_estimators: 100
  max_depth: 10
```

## 4. Train Locally

```bash
python train_local.py
```

This will:
- Load and preprocess data
- Train the model
- Log metrics and artifacts to MLflow
- Save model to `models/`

## 5. View MLflow Dashboard

```bash
mlflow ui
```

Open: http://localhost:5000

## 6. Train on AWS SageMaker (Optional)

### Prerequisites:
- AWS account configured: `aws configure`
- S3 bucket created
- SageMaker IAM role created

### Steps:
```bash
# Update config with your AWS details
# - Set role_arn in config.yaml
# - Set output_path S3 bucket

# Submit training job
python train_sagemaker.py --instance-type ml.m5.large
```

## 7. Make Predictions

```python
from src.inference import load_model, load_scaler, predict
import pandas as pd

model = load_model("models/model.pkl")
scaler = load_scaler("models/scaler.pkl")

# Load new data
new_data = pd.read_csv("new_data.csv")

# Make predictions
predictions = predict(model, scaler, new_data)
print(predictions)
```

## Project Structure

```
ml-sagemaker-mlflow/
├── config/config.yaml          ← Configuration
├── data/                        ← Data files
├── src/
│   ├── data_processing.py      ← Data pipeline
│   ├── model.py                ← Training logic
│   ├── evaluate.py             ← Metrics
│   └── inference.py            ← Predictions
├── sagemaker/
│   ├── train.py                ← SageMaker script
│   └── requirements.txt         ← Dependencies
├── models/                      ← Saved models
├── train_local.py              ← Local training entry
├── train_sagemaker.py          ← SageMaker submission
└── README.md
```

## Troubleshooting

### MLflow not tracking runs?
```bash
# Ensure tracking URI is set
mlflow ui
# Check if runs appear at http://localhost:5000
```

### Data loading fails?
- Verify CSV path in config.yaml
- Ensure CSV has proper columns
- Check for missing values

### SageMaker job fails?
- Verify AWS credentials: `aws sts get-caller-identity`
- Check S3 bucket exists and is accessible
- Review CloudWatch logs in SageMaker console

## Next Steps

- Add more models (XGBoost, LightGBM, etc.)
- Hyperparameter tuning with SageMaker Tuning Jobs
- Deploy model to SageMaker Endpoint
- Add data validation and monitoring
- Set up CI/CD pipeline
