import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os


def load_data(filepath):
    """Load CSV data."""
    print(f"Loading: {filepath}")
    return pd.read_csv(filepath)


def preprocess_data(df, target_column, test_size=0.2, random_state=42):
    """Load, scale, and split data."""
    print("Preprocessing...")
    
    df = df.dropna()  # Remove null values
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, scaler
