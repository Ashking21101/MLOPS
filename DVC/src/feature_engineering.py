import numpy as np
import pandas as pd
import os
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer

# UPDATED: Points to the project root (DVC/) instead of the src/ folder
BASE_DIR = Path(__file__).resolve().parent.parent

# fetch the data from data/processed (Relative to project root)
train_data = pd.read_csv(BASE_DIR / 'data' / 'processed' / 'train_processed.csv')
test_data = pd.read_csv(BASE_DIR / 'data' / 'processed' / 'test_processed.csv')

train_data.fillna('', inplace=True)
test_data.fillna('', inplace=True)

# apply BoW
X_train = train_data['content'].values
y_train = train_data['sentiment'].values

# Ensure labels are numeric (Mapping if not already handled in preprocessing)
label_map = {'happiness': 1, 'sadness': 0}
y_train = np.array([label_map.get(value, value) for value in y_train])

X_test = test_data['content'].values
y_test = test_data['sentiment'].values
y_test = np.array([label_map.get(value, value) for value in y_test])

# Apply Bag of Words (CountVectorizer)
vectorizer = CountVectorizer(max_features=50)

# Fit and transform
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

# Create dataframes for storage
train_df = pd.DataFrame(X_train_bow.toarray())
train_df['label'] = y_train

test_df = pd.DataFrame(X_test_bow.toarray())
test_df['label'] = y_test

# Store the data inside data/features (Relative to project root)
data_path = BASE_DIR / "data" / "features"
os.makedirs(data_path, exist_ok=True)

# Use index=False to keep CSVs clean for DVC tracking
train_df.to_csv(data_path / "train_bow.csv", index=False)
test_df.to_csv(data_path / "test_bow.csv", index=False)

print(f"Features successfully saved to: {data_path}")