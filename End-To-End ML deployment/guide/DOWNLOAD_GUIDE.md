# ML Project with MLflow & SageMaker - DOWNLOAD READY ✅

## 📦 ZIP File Contents

**File**: `ml-sagemaker-mlflow.zip` (17 KB)

### Quick Start (3 steps)
```bash
unzip ml-sagemaker-mlflow.zip
cd ml-sagemaker-mlflow
pip install -r requirements.txt
python generate_sample_data.py
python train_local.py
```

---

## 📂 What You Get

### Documentation (4 files)
- **README.md** - Overview & quick start
- **QUICKSTART.md** - Detailed setup guide
- **QUICK_REFERENCE.md** - Cheat sheet (THIS IS MOST USEFUL)
- **PROJECT_STRUCTURE.md** - Deep dive into architecture

### Code Files (SIMPLE & MINIMAL)

#### Entry Points (what you run)
- `train_local.py` (40 lines) - Train locally with MLflow
- `train_sagemaker.py` (40 lines) - Submit to AWS SageMaker
- `generate_sample_data.py` (40 lines) - Create test data
- `deploy_model.py` (15 lines) - Deploy to endpoint (optional)

#### Core Modules (src/ folder)
- `data_processing.py` (30 lines) - Load, clean, scale, split data
- `model.py` (20 lines) - Train model
- `evaluate.py` (25 lines) - Calculate metrics
- `inference.py` (20 lines) - Make predictions

#### Configuration
- `config/config.yaml` - All settings in one place
- `requirements.txt` - Dependencies (7 packages)
- `.gitignore` - Git ignore rules

#### SageMaker
- `sagemaker/train.py` (35 lines) - Training script for SageMaker
- `sagemaker/requirements.txt` - SageMaker dependencies

---

## 🎯 Total Code: ~200 Lines!

Everything is **simple**, **minimal**, and **easy to understand**.

### Sample File Sizes
- data_processing.py: 30 lines
- model.py: 20 lines
- evaluate.py: 25 lines
- inference.py: 20 lines
- train_local.py: 40 lines

No bloated code, no unnecessary complexity!

---

## 🚀 What It Does

### Local Training Flow
```
1. Load CSV data
2. Handle missing values
3. Scale features
4. Split train/test
5. Train RandomForest
6. Calculate metrics
7. Log everything to MLflow
8. Save model + scaler
```

### SageMaker Training Flow
```
Same as above, but:
- Runs on AWS infrastructure
- Saves to S3
- Scales to large datasets
- Optional MLflow tracking
```

---

## ⚙️ Technologies Included

```
MLflow          - Experiment tracking
scikit-learn    - Machine learning
pandas          - Data handling
boto3           - AWS integration
SageMaker       - Cloud training
```

---

## 📋 File Checklist

- ✅ README.md
- ✅ QUICKSTART.md
- ✅ QUICK_REFERENCE.md
- ✅ PROJECT_STRUCTURE.md
- ✅ requirements.txt
- ✅ config/config.yaml
- ✅ src/data_processing.py
- ✅ src/model.py
- ✅ src/evaluate.py
- ✅ src/inference.py
- ✅ train_local.py
- ✅ train_sagemaker.py
- ✅ generate_sample_data.py
- ✅ sagemaker/train.py
- ✅ deploy_model.py
- ✅ .gitignore

**Total: 16 files, ~200 lines of code, fully documented**

---

## 🎓 How to Use

### Step 1: Extract & Install
```bash
unzip ml-sagemaker-mlflow.zip
cd ml-sagemaker-mlflow
pip install -r requirements.txt
```

### Step 2: Generate Test Data (Optional)
```bash
python generate_sample_data.py
# Creates: data/raw/data.csv
```

### Step 3: Train Locally
```bash
python train_local.py
# Trains model, saves artifacts, logs to MLflow
```

### Step 4: View Results
```bash
mlflow ui
# Open: http://localhost:5000
# See: experiments, metrics, artifacts
```

### Step 5: Use Your Own Data
```bash
# Place your CSV in: data/raw/your_data.csv
# Edit: config/config.yaml
#   - Update raw_path
#   - Update target_column
# Run: python train_local.py
```

---

## 🔄 Workflow

```
1. Data (CSV)
       ↓
2. Data Processing (clean, scale, split)
       ↓
3. Model Training (RandomForest)
       ↓
4. Evaluation (metrics)
       ↓
5. Save Artifacts
       ↓
6. MLflow Tracking
       ↓
7. Done! (model.pkl, scaler.pkl, metrics.json)
```

---

## 🌐 Local vs SageMaker

### Local Training
- Fast (minutes)
- Good for development
- Limited to your machine
- `python train_local.py`

### SageMaker Training
- Scalable (hours)
- Production ready
- Use large datasets
- `python train_sagemaker.py`

---

## ✨ Key Features

✅ **Simple Code** - ~200 lines, easy to modify
✅ **MLflow Integration** - Automatic experiment tracking
✅ **Dual Training** - Local + Cloud (AWS SageMaker)
✅ **Configuration-Driven** - YAML-based settings
✅ **Modular Design** - Reusable components
✅ **Well Documented** - 4 guide files included
✅ **Production Ready** - Real ML pipeline
✅ **Easy to Extend** - Add models, preprocessing, etc.

---

## 📊 Example Output

After running `python train_local.py`:

```
Loading: data/raw/data.csv
Preprocessing...
Train: (800, 10), Test: (200, 10)
Training model...
Done!

Metrics:
  accuracy: 0.8750
  precision: 0.8667
  recall: 0.8750
  f1: 0.8708

✓ Complete! Run ID: abc123def456

MLflow Dashboard: http://localhost:5000
Models saved to: models/
```

---

## 🔧 Customization

### Change Model Type
Edit `src/model.py`:
```python
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(n_estimators=100)
```

### Add Preprocessing
Edit `src/data_processing.py`:
```python
# Add feature engineering, handle outliers, etc.
```

### Adjust Parameters
Edit `config/config.yaml`:
```yaml
model:
  n_estimators: 200  # More trees
  max_depth: 15      # Deeper trees
```

---

## ❓ FAQ

**Q: Do I need AWS?**
A: No! Local training works standalone. AWS is optional for scaling.

**Q: Can I use my own data?**
A: Yes! CSV file with features + target column.

**Q: What if I want to use XGBoost?**
A: Change `src/model.py` - just swap imports and parameters.

**Q: How do I deploy?**
A: Use `deploy_model.py` or SageMaker endpoints (optional).

**Q: What's the learning curve?**
A: Low! Code is simple and well-commented. ~2 hours to understand.

---

## 📚 Documentation Files Guide

| File | Read When | Time |
|------|-----------|------|
| **README.md** | First | 2 min |
| **QUICK_REFERENCE.md** | For quick lookup | 5 min |
| **QUICKSTART.md** | Detailed setup | 10 min |
| **PROJECT_STRUCTURE.md** | Deep understanding | 15 min |

---

## 🎯 Next Steps After Download

1. ✅ Extract ZIP
2. ✅ Install dependencies
3. ✅ Read README.md
4. ✅ Run generate_sample_data.py
5. ✅ Run train_local.py
6. ✅ Open MLflow UI
7. ✅ Replace with your data
8. ✅ Customize config.yaml
9. ✅ Train on your dataset
10. ✅ Explore MLflow results

---

## 🚀 You're Ready!

Download the ZIP, follow the 5-minute setup, and you have a working ML + MLflow + SageMaker project!

**Questions?** Check QUICK_REFERENCE.md - it has all the answers!

---

**Happy Machine Learning! 🎉**
