## Removed stray top-level pickle.dump line that caused NameError
import pandas as pd
import numpy as np
import pickle
import yaml
import logging
from typing import Any, Dict, Tuple
from sklearn.ensemble import RandomForestClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def load_params(path: str) -> Dict[str, Any]:
    """Load parameters from a YAML file."""
    try:
        with open(path, "r") as file:
            params = yaml.safe_load(file)
        logging.info(f"Loaded parameters from {path}")
        return params
    except Exception as e:
        logging.error(f"Error loading parameters from {path}: {e}")
        raise

def load_data(path: str) -> pd.DataFrame:
    """Load training data from CSV."""
    try:
        df = pd.read_csv(path)
        logging.info(f"Loaded training data from {path} with shape {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading training data from {path}: {e}")
        raise

def prepare_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare features and labels from DataFrame."""
    try:
        x_train = df.drop(columns=['label']).values
        y_train = df['label'].values
        logging.info(f"Prepared features and labels. Features shape: {x_train.shape}, Labels shape: {y_train.shape}")
        return x_train, y_train
    except Exception as e:
        logging.error(f"Error preparing features and labels: {e}")
        raise

def train_model(x_train: np.ndarray, y_train: np.ndarray, n_estimators: int, max_depth: int) -> RandomForestClassifier:
    """Train RandomForest model."""
    try:
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(x_train, y_train)
        logging.info("RandomForest model trained successfully.")
        return model
    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise

def save_model(model: RandomForestClassifier, path: str) -> None:
    """Save trained model to disk using pickle."""
    try:
        with open(path, "wb") as f:
            pickle.dump(model, f)
        logging.info(f"Model saved to {path}")
    except Exception as e:
        logging.error(f"Error saving model to {path}: {e}")
        raise

def main() -> None:
    try:
        params = load_params("params.yaml")
        n_estimators = params["model_building"]["n_estimators"]
        max_depth = params["model_building"]["max_depth"]
        train_data = load_data("data/interim/train_tfidf.csv")
        x_train, y_train = prepare_features(train_data)
        model = train_model(x_train, y_train, n_estimators, max_depth)
        save_model(model, "models/random_forest_model.pkl")
        logging.info("Model building completed successfully.")
    except Exception as e:
        logging.critical(f"Model building failed: {e}")

if __name__ == "__main__":
    main()