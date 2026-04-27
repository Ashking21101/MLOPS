import boto3
import pandas as pd
import io
import json

def test_full_csv(endpoint_name, file_path):
    # 1. Initialize AWS Client
    runtime = boto3.client('sagemaker-runtime', region_name='ap-south-1')
    
    # 2. Load and Prepare Data
    df = pd.read_csv(file_path)
    print(f"✓ Loaded {len(df)} rows from {file_path}")
    
    # CRITICAL: Drop the target column (the last one) so we send exactly 10 features
    X = df.iloc[:, :-1] 
    print(f"✓ Verified: Sending {X.shape[1]} features to model.")

    # 3. Convert entire dataframe to CSV string
    csv_buffer = io.StringIO()
    X.to_csv(csv_buffer, index=False, header=False)
    payload = csv_buffer.getvalue()

    print(f"→ Sending full payload to {endpoint_name}...")

    try:
        # 4. Invoke Endpoint
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='text/csv',
            Body=payload
        )

        # 5. Parse and Display All Predictions
        result = json.loads(response['Body'].read().decode())
        predictions = result['predictions']
        
        print("\n" + "="*30)
        print(f"TOTAL PREDICTIONS RECEIVED: {len(predictions)}")
        print(f"FIRST 10 RESULTS: {predictions[:10]}")
        print("="*30)
        
        # Optional: Add predictions back to your dataframe to view them side-by-side
        df['model_predictions'] = predictions
        df.to_csv("data/raw/final_results.csv", index=False)
        print("✓ Full results saved to data/raw/final_results.csv")

    except Exception as e:
        print(f"✗ Error during full inference: {str(e)}")

if __name__ == "__main__":
    MY_ENDPOINT = "sagemaker-scikit-learn-2026-04-27-16-13-13-372"
    test_full_csv(MY_ENDPOINT, "data/raw/sample_data.csv")