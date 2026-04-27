# Wine Quality Prediction - MLOps on AWS SageMaker

An end-to-end Machine Learning pipeline using Scikit-Learn, MLflow, and AWS SageMaker.

## Project Structure
- `src/`: Core logic for data processing and model training.
- `config/`: Configuration files for environment and hyper-parameters.
- `sagemaker_serving/`: Inference script used by the SageMaker endpoint.
- `test_endpoint_2.py`: Client-side script to verify the live production API.

## Features
- **Experiment Tracking:** Integrated with MLflow.
- **Cloud Deployment:** Deployed as a native AWS SageMaker endpoint.
- **Scalable Inference:** Supports batch CSV processing via a 10-feature schema.