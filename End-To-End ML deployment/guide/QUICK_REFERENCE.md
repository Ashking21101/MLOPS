# Quick Reference Guide

## 📦 What's in the ZIP

```
ml-sagemaker-mlflow/
├── README.md                   ← Start here
├── QUICKSTART.md               ← Setup guide
├── PROJECT_STRUCTURE.md        ← Detailed info
│
├── config/config.yaml          ← Edit: paths, model params
├── requirements.txt            ← Dependencies
│
├── src/                        ← Core code (simple!)
│   ├── data_processing.py      ← 30 lines
│   ├── model.py                ← 20 lines
│   ├── evaluate.py             ← 25 lines
│   └── inference.py            ← 20 lines
│
├── train_local.py              ← 40 lines
├── train_sagemaker.py          ← 40 lines
├── generate_sample_data.py      ← 40 lines
└── sagemaker/train.py          ← 35 lines
```

## 🚀 5-Minute Setup

```bash
# 1. Unzip
unzip ml-sagemaker-mlflow.zip
cd ml-sagemaker-mlflow

# 2. Install
pip install -r requirements.txt

# 3. Generate test data
python generate_sample_data.py

# 4. Train
python train_local.py

# 5. View results
mlflow ui
# Open: http://localhost:5000
```

## 📊 What Gets Created

```
models/
  ├── model.pkl        ← Trained model
  ├── scaler.pkl       ← Feature scaler
  └── metrics.json     ← Performance metrics

mlruns/
  └── [experiment data] ← MLflow tracking

data/processed/
  └── [processed files] ← (if you want to save)
```

## 🔧 Edit These 3 Files Only

### 1. `config/config.yaml`
```yaml
data:
  raw_path: "data/raw/YOUR_DATA.csv"  ← Your CSV file
  target_column: "target"              ← Your target column

model:
  n_estimators: 100
  max_depth: 10
```

### 2. `src/data_processing.py`
- Customize preprocessing if needed
- Add feature engineering, scaling, etc.

### 3. `src/model.py`
- Change to XGBoost, LightGBM, etc.
- Adjust hyperparameters

## 🎯 Common Tasks

### Add your own data
```bash
# Place CSV in:
data/raw/your_data.csv

# Update config.yaml:
raw_path: "data/raw/your_data.csv"
target_column: "your_target_column"

# Run:
python train_local.py
```

### Change model type
```python
# In src/model.py, replace RandomForest with:

from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(n_estimators=100)
```

### Add preprocessing
```python
# In src/data_processing.py, add:

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X = poly.fit_transform(X)
```

### View MLflow results
```bash
mlflow ui
# Then open: http://localhost:5000
# See: experiments, runs, metrics, artifacts
```

## 🌐 Train on AWS SageMaker

```bash
# 1. Configure AWS
aws configure
# Enter: Access Key, Secret Key, Region, Format

# 2. Create S3 bucket
# AWS Console → S3 → Create bucket

# 3. Create SageMaker role
# AWS Console → IAM → Create role with SageMaker policy

# 4. Update config.yaml
sagemaker:
  role_arn: "arn:aws:iam::ACCOUNT:role/SageMakerRole"
  output_path: "s3://YOUR_BUCKET/output"

# 5. Submit job
python train_sagemaker.py --instance-type ml.m5.large
```

## 📁 File Purposes

| File | Lines | Purpose |
|------|-------|---------|
| `data_processing.py` | 30 | Load CSV → Remove nulls → Scale → Split |
| `model.py` | 20 | Train RandomForest |
| `evaluate.py` | 25 | Calculate accuracy, precision, recall, F1 |
| `inference.py` | 20 | Load model → Make predictions |
| `train_local.py` | 40 | Main entry: local training with MLflow |
| `train_sagemaker.py` | 40 | Submit job to SageMaker |
| `generate_sample_data.py` | 40 | Generate test data |

## ✅ Code is Simple!

- **Minimal imports** - Only necessary libraries
- **Clear function names** - Easy to understand
- **Single responsibility** - Each function does one thing
- **Good comments** - Explain what's happening
- **~200 lines total** - Not a lot of code!

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| "No module named 'src'" | Run from project root: `cd ml-sagemaker-mlflow` |
| "CSV not found" | Check `config.yaml` raw_path matches your file |
| "MLflow shows no runs" | Run `mlflow ui` in another terminal |
| AWS error | Check: `aws sts get-caller-identity` |

## 🎓 Learning Path

1. **Day 1**: Run `generate_sample_data.py` → `train_local.py` → View MLflow
2. **Day 2**: Edit `config.yaml` with your own data → Train
3. **Day 3**: Modify `src/data_processing.py` for better preprocessing
4. **Day 4**: Try different models in `src/model.py`
5. **Day 5**: Deploy to SageMaker

## 📞 Next Steps

- [ ] Add cross-validation
- [ ] Add feature importance plots
- [ ] Try hyperparameter tuning
- [ ] Add data validation
- [ ] Create inference API (Flask/FastAPI)
- [ ] Deploy to SageMaker Endpoint
- [ ] Set up CI/CD pipeline

---

**That's it! Simple ML + MLflow + SageMaker ready to go! 🚀**
