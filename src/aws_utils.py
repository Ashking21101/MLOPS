import yaml


def load_config(config_path="config/config.yaml"):
    """Load project configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_sagemaker_role(config, role_arn=None):
    """Return the SageMaker execution role ARN from CLI or config."""
    role = role_arn or config["sagemaker"].get("role_arn")
    if not role or "YOUR_ACCOUNT_ID" in role:
        raise ValueError(
            "Set sagemaker.role_arn in config/config.yaml or pass --role-arn "
            "with an IAM role ARN, for example "
            "arn:aws:iam::<account-id>:role/<sagemaker-execution-role>."
        )
    return role
