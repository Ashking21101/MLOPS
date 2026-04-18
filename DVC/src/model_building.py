import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier

# UPDATED: Points to the project root (DVC/) instead of the src/ folder
BASE_DIR = Path(__file__).resolve().parent.parent

# fetch the data from data/features (Relative to project root)
train_data_path = BASE_DIR / 'data' / 'features' / 'train_bow.csv'
train_data = pd.read_csv(train_data_path)

X_train = train_data.iloc[:, 0:-1].values
y_train = train_data.iloc[:, -1].values

# Define and train the Gradient Boosting model
clf = GradientBoostingClassifier(n_estimators=50)
clf.fit(X_train, y_train)

# Save the model directly to the project root
model_path = BASE_DIR / 'model.pkl'

with open(model_path, 'wb') as f:
    pickle.dump(clf, f)

print(f"Model successfully saved to: {model_path}")