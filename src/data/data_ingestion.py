import numpy as np
import pandas as pd
import os
import yaml
import logging
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def load_params(path: str) -> Dict[str, Any]:
    """Load parameters from a YAML file."""
    try:
        with open(path, 'r') as file:
            params = yaml.safe_load(file)
        logging.info(f"Loaded parameters from {path}")
        return params
    except Exception as e:
        logging.error(f"Error loading parameters from {path}: {e}")
        raise

def fetch_dataset(url: str) -> pd.DataFrame:
    """Fetch dataset from a given URL."""
    try:
        df = pd.read_csv(url)
        logging.info(f"Dataset loaded from {url} with shape {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading dataset from {url}: {e}")
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the dataset for analysis."""
    try:
        df = df.drop(columns=['tweet_id'])
        filtered_df = df[df['sentiment'].isin(['happiness', 'sadness'])].copy()
        filtered_df['sentiment'] = filtered_df['sentiment'].replace({'happiness': 1, 'sadness': 0})
        logging.info(f"Preprocessed data shape: {filtered_df.shape}")
        return filtered_df
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        raise

def split_and_save_data(df: pd.DataFrame, test_size: float, train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train/test and save to CSV files."""
    try:
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        logging.info(f"Train data saved to {train_path} ({train_data.shape})")
        logging.info(f"Test data saved to {test_path} ({test_data.shape})")
        return train_data, test_data
    except Exception as e:
        logging.error(f"Error splitting/saving data: {e}")
        raise

def main() -> None:
    """Main function to orchestrate data ingestion."""
    try:
        params = load_params('params.yaml')
        test_size = params['data_ingestion']['test_size']
        url = 'https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv'
        df = fetch_dataset(url)
        final_df = preprocess_data(df)
        split_and_save_data(final_df, test_size, 'data/raw/train.csv', 'data/raw/test.csv')
        logging.info("Data ingestion completed successfully.")
    except Exception as e:
        logging.critical(f"Data ingestion failed: {e}")

if __name__ == "__main__":
    main()