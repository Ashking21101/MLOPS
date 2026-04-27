from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json


def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)
    
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    }
    
    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    return metrics, y_pred


def save_metrics(metrics, filepath="models/metrics.json"):
    """Save metrics to file."""
    with open(filepath, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: {filepath}")
