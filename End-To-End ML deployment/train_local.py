import mlflow
import yaml
import os
import joblib
from src.data_processing import load_data, preprocess_data
from src.model import train_model, save_model
from src.evaluate import evaluate_model, save_metrics
from src.mlflow_utils import set_or_restore_experiment


def load_config():
    """Load config from YAML."""
    with open("config/config.yaml", "r") as f:
        return yaml.safe_load(f)


def main():
    """Train model with MLflow tracking."""
    config = load_config()
    
    # Set experiment
    set_or_restore_experiment(config["mlflow"]["experiment_name"])
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(config["model"])
        
        # Load data
        df = load_data(config["data"]["raw_path"])
        X_train, X_test, y_train, y_test, scaler = preprocess_data(
            df, config["data"]["target_column"]
        )
        
        # Train
        model = train_model(X_train, y_train, config["model"])
        
        # Evaluate
        metrics, _ = evaluate_model(model, X_test, y_test)
        
        # Log metrics
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        
        # Save
        os.makedirs("models", exist_ok=True)
        save_model(model)
        joblib.dump(scaler, "models/scaler.pkl")
        save_metrics(metrics)
        
        # Log artifacts
        mlflow.log_artifact("models/model.pkl")
        mlflow.log_artifact("models/scaler.pkl")
        
        print(f"\n✓ Complete! Run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
