from sklearn.ensemble import RandomForestClassifier
import joblib
import os


def train_model(X_train, y_train, model_config):
    """Train RandomForest model."""
    print("Training model...")
    
    model = RandomForestClassifier(
        n_estimators=model_config.get("n_estimators", 100),
        max_depth=model_config.get("max_depth", 10),
        random_state=model_config.get("random_state", 42),
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("Done!")
    return model


def save_model(model, filepath="models/model.pkl"):
    """Save model to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Saved: {filepath}")
