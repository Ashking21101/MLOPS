"""Deploy model to SageMaker Endpoint"""
import argparse
import sagemaker
from sagemaker.sklearn.model import SKLearnModel
from src.aws_utils import get_sagemaker_role, load_config

def deploy(model_path=None, instance_type=None, role_arn=None):
    """Deploy model to SageMaker."""
    config = load_config()
    role = get_sagemaker_role(config, role_arn)
    instance_type = instance_type or "ml.t2.medium" # Default to cheaper instance for hosting

    if not model_path or not model_path.endswith("model.tar.gz"):
        raise ValueError(
            "Pass the full S3 model artifact path ending in model.tar.gz, for example "
            "s3://bucket/output/ml-training-1234567890/output/model.tar.gz"
        )

    print("="*50)
    print("Deploying Model to SageMaker Endpoint")
    print(f"S3 Model Path: {model_path}")
    print(f"Instance: {instance_type}")
    print(f"Role: {role}")
    print("="*50)
    
    model = SKLearnModel(
        model_data=model_path,
        role=role,
        entry_point="inference.py",
        source_dir="sagemaker_serving",
        framework_version="1.0-1"
    )
    
    print("Starting deployment (this will take ~10 minutes)...")
    
    predictor = model.deploy(
        instance_type=instance_type,
        initial_instance_count=1
    )
    
    print(f"Successfully deployed! Endpoint name: {predictor.endpoint_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # When running, pass the S3 path of your model.tar.gz here
    parser.add_argument("--model-path", type=str, required=True, help="S3 path to model.tar.gz")
    parser.add_argument("--instance-type", type=str, default="ml.t2.medium")
    parser.add_argument("--role-arn", type=str, default=None)
    
    args = parser.parse_args()
    deploy(args.model_path, args.instance_type, args.role_arn)
