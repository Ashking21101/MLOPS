import mlflow
from mlflow.tracking import MlflowClient


def set_or_restore_experiment(experiment_name):
    """Set an MLflow experiment, restoring it first if it was soft-deleted."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment and experiment.lifecycle_stage == "deleted":
        client.restore_experiment(experiment.experiment_id)
        print(f"Restored deleted MLflow experiment: {experiment_name}")

    mlflow.set_experiment(experiment_name)
