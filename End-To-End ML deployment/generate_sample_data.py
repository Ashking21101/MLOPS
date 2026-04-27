import pandas as pd
import numpy as np
import argparse
import os


def generate_data(n_rows=1000, n_features=10):
    """Generate synthetic classification data."""
    np.random.seed(42)
    
    X = np.random.randn(n_rows, n_features)
    y = (X[:, 0] + X[:, 1] * 0.5 - X[:, 2] * 0.3 > 0).astype(int)
    
    # Add noise
    flip_idx = np.random.choice(n_rows, size=int(0.1 * n_rows), replace=False)
    y[flip_idx] = 1 - y[flip_idx]
    
    cols = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    df['target'] = y
    
    return df


def main(n_rows=1000, output_path="data/raw/data.csv"):
    """Generate and save data."""
    print(f"Generating {n_rows} samples...")
    df = generate_data(n_rows=n_rows)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"✓ Saved: {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--output", type=str, default="data/raw/data.csv")
    
    args = parser.parse_args()
    main(args.rows, args.output)
