import sagemaker
import argparse
from sagemaker.sklearn.estimator import SKLearn
import time
from src.aws_utils import get_sagemaker_role, load_config


def submit_job(instance_type=None, instance_count=None, role_arn=None):
    """Submit training job to SageMaker."""
    config = load_config()
    
    session = sagemaker.Session()
    role = get_sagemaker_role(config, role_arn)
    instance_type = instance_type or config["sagemaker"]["instance_type"]
    instance_count = instance_count or config["sagemaker"]["instance_count"]
    
    print("="*50)
    print("Submitting SageMaker Training Job")
    print(f"Instance: {instance_type} x{instance_count}")
    print(f"Role: {role}")
    print("="*50)
    
    # Create estimator
    estimator = SKLearn(
        entry_point="train.py",
        source_dir="sagemaker",
        dependencies=["src", "config"],
        role=role,
        instance_type=instance_type,
        instance_count=instance_count,
        framework_version="1.0-1",
        py_version="py3",
        script_mode=True,
        output_path=config["sagemaker"]["output_path"]
    )
    
    # Upload data
    data_path = session.upload_data("data/raw", key_prefix="training-data")
    print(f"Data uploaded: {data_path}")
    
    # Submit job
    job_name = f"{config['sagemaker']['job_name_prefix']}-{int(time.time())}"
    estimator.fit({"train": data_path}, job_name=job_name)
    
    print(f"\n✓ Job submitted: {job_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-type", type=str, default=None)
    parser.add_argument("--instance-count", type=int, default=None)
    parser.add_argument("--role-arn", type=str, default=None)
    
    args = parser.parse_args()
    submit_job(args.instance_type, args.instance_count, args.role_arn)
