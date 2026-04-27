import json
import os

import joblib
import numpy as np
import pandas as pd


def _feature_frame(data):
    df = pd.DataFrame(data).astype(float)
    df.columns = [f"feature_{i}" for i in range(df.shape[1])]
    return df


def model_fn(model_dir):
    """Load model artifacts for SageMaker hosting."""
    model = joblib.load(os.path.join(model_dir, "model.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    return {"model": model, "scaler": scaler}


def input_fn(request_body, request_content_type):
    """Parse CSV or JSON request bodies."""
    if request_content_type == "text/csv":
        rows = [line.split(",") for line in request_body.strip().splitlines()]
        return _feature_frame(rows)

    if request_content_type == "application/json":
        payload = json.loads(request_body)
        data = payload.get("instances", payload)
        return _feature_frame(data)

    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, artifacts):
    """Scale inputs and return model predictions."""
    X_scaled = pd.DataFrame(
        artifacts["scaler"].transform(input_data),
        columns=input_data.columns,
    )
    return artifacts["model"].predict(X_scaled)


def output_fn(prediction, accept):
    """Serialize predictions."""
    values = np.asarray(prediction).tolist()

    if accept == "text/csv":
        return ",".join(str(value) for value in values), accept

    return json.dumps({"predictions": values}), "application/json"
