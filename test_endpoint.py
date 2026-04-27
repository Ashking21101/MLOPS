import boto3
import json

# The name from your screenshot
ENDPOINT_NAME = "sagemaker-scikit-learn-2026-04-27-16-13-13-372"

runtime = boto3.client('sagemaker-runtime', region_name='ap-south-1')

# Replace this with 10 values that match your model's expected input features
# (I noticed your training logs said "Train: (800, 10)", so it expects 10 columns)
payload = {
    "instances": [
        [0.3, 0.1, 0.8, 0.9, 0.1, 0.2, 0.6, 0.7, 0.1, 1.0] 
    ]
}

print(f"Sending request to {ENDPOINT_NAME}...")

try:
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType='application/json',
        Body=json.dumps(payload)
    )
    
    result = json.loads(response['Body'].read().decode())
    print("-" * 30)
    print("PREDICTION SUCCESSFUL!")
    print(f"Result: {result}")
    print("-" * 30)
except Exception as e:
    print(f"Error: {e}")