# ML Project with MLflow & AWS SageMaker - Complete Overview

## 📁 Project Structure (Simplified)

```
ml-sagemaker-mlflow/
│
├── 📄 README.md                    # Project overview
├── 📄 QUICKSTART.md                # Getting started guide
├── 📄 requirements.txt             # Python dependencies
├── 📄 .gitignore                   # Git ignore rules
│
├── 📁 config/
│   └── 📄 config.yaml              # All configurations (data, model, AWS)
│
├── 📁 data/
│   ├── 📁 raw/                     # Raw data (your CSV files)
│   └── 📁 processed/               # Preprocessed data (auto-generated)
│
├── 📁 src/                         # Core application code
│   ├── 📄 __init__.py
│   ├── 📄 data_processing.py       # Load, clean, split data
│   ├── 📄 model.py                 # Train RandomForest
│   ├── 📄 evaluate.py              # Calculate metrics
│   └── 📄 inference.py             # Make predictions
│
├── 📁 sagemaker/                   # AWS SageMaker specific
│   ├── 📄 train.py                 # Training script for SageMaker
│   └── 📄 requirements.txt          # Dependencies for SageMaker
│
├── 📁 models/                      # Saved artifacts (auto-generated)
│   ├── 📄 model.pkl                # Trained model
│   ├── 📄 scaler.pkl               # Feature scaler
│   └── 📄 metrics.json             # Performance metrics
│
├── 📁 mlruns/                      # MLflow tracking (auto-generated)
│   └── [experiment runs...]
│
├── 📄 train_local.py               # Entry: Train locally with MLflow
├── 📄 train_sagemaker.py           # Entry: Submit job to SageMaker
└── 📄 deploy_model.py              # Entry: Deploy to SageMaker Endpoint
```

## 🔄 Workflow

### Option 1: Train Locally (Recommended for Development)

```
1. Prepare data → Place CSV in data/raw/
2. Configure → Edit config/config.yaml
3. Train → python train_local.py
4. Track → mlflow ui (view dashboard at localhost:5000)
5. Evaluate → Check metrics in MLflow
6. Predict → Use src/inference.py
```

### Option 2: Train on AWS SageMaker (For Production Scale)

```
1. Prepare data → Upload to S3
2. Configure → Set AWS credentials & S3 path in config.yaml
3. Submit → python train_sagemaker.py --instance-type ml.m5.large
4. Monitor → Check SageMaker console
5. Track → View MLflow (optional: remote tracking URI)
6. Deploy → python deploy_model.py
```

## 📊 What Each File Does

### Core Training Files

| File | Purpose |
|------|---------|
| `train_local.py` | Main entry point for local training with MLflow tracking |
| `train_sagemaker.py` | Submit training job to AWS SageMaker |
| `deploy_model.py` | Deploy trained model to SageMaker Endpoint |

### Source Code (src/)

| Module | Purpose |
|--------|---------|
| `data_processing.py` | Load CSV → Handle nulls → Scale features → Train/test split |
| `model.py` | Train RandomForestClassifier with config params |
| `evaluate.py` | Calculate accuracy, precision, recall, F1 score |
| `inference.py` | Load model/scaler → Make predictions on new data |

### Configuration

| File | Purpose |
|------|---------|
| `config/config.yaml` | Single source of truth for model, training, AWS, data paths |

### MLflow Integration

```python
# What gets logged automatically:
✓ Model parameters (n_estimators, max_depth, etc.)
✓ Training parameters (test_size, random_state, etc.)
✓ Metrics (accuracy, precision, recall, f1)
✓ Artifacts (model.pkl, scaler.pkl, metrics.json)
✓ Run ID for reproducibility
```

## 🚀 Quick Usage

### 1. Initial Setup (5 min)
```bash
cd ml-sagemaker-mlflow
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Add Your Data
```bash
# Create CSV file with features + target column
# Place in: data/raw/your_data.csv
```

### 3. Update Config
```bash
# Edit config/config.yaml
# - Update raw_path
# - Update target_column
# - Adjust model params if needed
```

### 4. Train Locally
```bash
python train_local.py
```

### 5. View Results
```bash
mlflow ui
# Open http://localhost:5000
```

## 🎯 Key Features

✅ **MLflow Integration** - Automatic experiment tracking
✅ **Modular Code** - Reusable components (data, model, eval)
✅ **Simple Config** - YAML-based, easy to modify
✅ **Local & Cloud** - Train locally or on SageMaker
✅ **Artifact Management** - Models, scalers, metrics saved
✅ **Clean Separation** - Data → Model → Evaluation → Inference

## 📝 Configuration Guide (config.yaml)

```yaml
# Which model to use
model:
  type: "RandomForest"
  n_estimators: 100        # Number of trees
  max_depth: 10            # Max tree depth
  random_state: 42         # For reproducibility

# Data splitting
training:
  test_size: 0.2           # 80/20 split
  random_state: 42

# AWS settings (for SageMaker)
sagemaker:
  region: "ap-south-1"
  role_arn: "arn:aws:iam::ACCOUNT:role/SageMakerRole"
  instance_type: "ml.m5.large"
  output_path: "s3://your-bucket/output"

# Data paths
data:
  raw_path: "data/raw/data.csv"
  target_column: "target"

# MLflow tracking
mlflow:
  experiment_name: "ml-sagemaker-experiment"
```

## 🔐 AWS Setup (SageMaker Only)

If using SageMaker, you need:

1. **AWS Account** with credentials configured:
   ```bash
   aws configure
   # Enter: Access Key ID, Secret Access Key, Region, Output format
   ```

2. **SageMaker IAM Role**:
   ```bash
   # AWS Console → IAM → Create role with AmazonSageMakerFullAccess
   # Copy role ARN to config.yaml
   ```

3. **S3 Bucket**:
   ```bash
   # AWS Console → S3 → Create bucket
   # Update output_path in config.yaml
   ```

## 📊 Example Metrics Output

After training, you'll see:
```
Evaluation Metrics:
  accuracy: 0.8742
  precision: 0.8650
  recall: 0.8750
  f1: 0.8700

Artifacts saved:
  ✓ models/model.pkl
  ✓ models/scaler.pkl
  ✓ models/metrics.json
```

## 🔮 Next Steps / Extensions

### Easy Additions
- [ ] Hyperparameter tuning (GridSearchCV)
- [ ] Cross-validation scoring
- [ ] Feature importance plots
- [ ] Data validation checks
- [ ] Model versioning

### Medium Additions
- [ ] Support multiple model types (XGBoost, LightGBM)
- [ ] Automated model comparison
- [ ] Production inference API (Flask/FastAPI)
- [ ] Model monitoring dashboards

### Advanced Additions
- [ ] SageMaker hyperparameter tuning jobs
- [ ] Automated retraining pipelines
- [ ] A/B testing framework
- [ ] Data drift detection
- [ ] Model serving with Lambda

## ⚠️ Common Issues & Fixes

| Issue | Solution |
|-------|----------|
| "No module named 'src'" | Run from project root: `cd ml-sagemaker-mlflow` |
| "config.yaml not found" | Ensure config/config.yaml exists |
| "Data file not found" | Check raw_path in config.yaml matches your file |
| "MLflow not tracking" | Run `mlflow ui` in separate terminal |
| SageMaker job fails | Check AWS credentials: `aws sts get-caller-identity` |

## 📞 Support

For debugging:
1. Check CloudWatch logs (SageMaker jobs)
2. Review MLflow UI for run details
3. Verify file paths in config.yaml
4. Ensure AWS credentials are correct

---

**This is a minimal, production-ready template. Customize as needed!**
