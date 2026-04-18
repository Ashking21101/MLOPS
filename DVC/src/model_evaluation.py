import numpy as np
import pandas as pd
import pickle
import json
import os
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# UPDATED: Points to the project root (DVC/) instead of the src/ folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Load the model from the project root
model_path = BASE_DIR / 'model.pkl'
with open(model_path, 'rb') as f:
    clf = pickle.load(f)

# Fetch the test data from data/features (Relative to project root)
test_data_path = BASE_DIR / 'data' / 'features' / 'test_bow.csv'
test_data = pd.read_csv(test_data_path)

X_test = test_data.iloc[:, 0:-1].values
y_test = test_data.iloc[:, -1].values

# Ensure labels are numeric
label_map = {'happiness': 1, 'sadness': 0}
y_test = np.array([label_map.get(value, value) for value in y_test])

# Predictions
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]

# Calculate evaluation metrics
metrics_dict = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred),
    'auc': roc_auc_score(y_test, y_pred_proba)
}

# Save metrics to the project root
metrics_path = BASE_DIR / 'metrics.json'
with open(metrics_path, 'w') as file:
    json.dump(metrics_dict, file, indent=4)

print(f"Metrics successfully saved to: {metrics_path}")