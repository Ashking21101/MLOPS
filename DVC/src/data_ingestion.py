import numpy as np
import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import yaml

# UPDATED: Points to the Project Root (one level up from src/)
BASE_DIR = Path(__file__).resolve().parent.parent


def load_data(data_url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_url)
        return df
    except pd.errors.ParserError as e:
        print(f"Error: Failed to parse the CSV file from {data_url}.")
        raise
    except Exception as e:
        print(f"Error: An unexpected error occurred while loading the data.")
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.drop(columns=['tweet_id'])
        final_df = df.loc[df['sentiment'].isin(['happiness', 'sadness'])].copy()
        final_df['sentiment'] = final_df['sentiment'].replace({'happiness': 1, 'sadness': 0})
        return final_df
    except KeyError as e:
        print(f"Error: Missing column {e} in the dataframe.")
        raise
    except Exception as e:
        print(f"Error: An unexpected error occurred during preprocessing.")
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    try:
        # This will now create /DVC/data/raw instead of /DVC/src/data/raw
        final_data_path = BASE_DIR / data_path / 'raw'
        final_data_path.mkdir(parents=True, exist_ok=True)
        
        train_data.to_csv(final_data_path / "train.csv", index=False)
        test_data.to_csv(final_data_path / "test.csv", index=False)
        print(f"Data successfully saved to {final_data_path}")
    except Exception as e:
        print(f"Error: An unexpected error occurred while saving the data.")
        raise


def main():
    try:
        # Use the raw URL for the dataset
        df = load_data(data_url='https://raw.githubusercontent.com/entbappy/Branching-tutorial/refs/heads/master/tweet_emotions.csv')
        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size=0.2, random_state=42)
        
        # 'data' here refers to the folder name in the project root
        save_data(train_data, test_data, data_path='data')
    except Exception as e:
        print(f"Error: {e}")
        print("Failed to complete the data ingestion process.")


if __name__ == '__main__':
    main()