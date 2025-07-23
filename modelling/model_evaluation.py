from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import pandas as pd
import pickle
import json
import logging
from typing import Dict, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def load_model(path: str) -> Any:
    """Load a trained model from disk."""
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        logging.info(f"Model loaded from {path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model from {path}: {e}")
        raise

def load_test_data(path: str) -> Tuple[pd.DataFrame, Any, Any]:
    """Load test data and split into features and labels."""
    try:
        df = pd.read_csv(path)
        X_test = df.drop(columns=['label']).values
        y_test = df['label'].values
        logging.info(f"Test data loaded from {path} with shape {df.shape}")
        return df, X_test, y_test
    except Exception as e:
        logging.error(f"Error loading test data from {path}: {e}")
        raise

def evaluate_model(model: Any, X_test: Any, y_test: Any) -> Dict[str, float]:
    """Evaluate model and return metrics."""
    try:
        y_pred = model.predict(X_test)
        metrics_dict = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred)
        }
        logging.info(f"Evaluation metrics: {metrics_dict}")
        return metrics_dict
    except Exception as e:
        logging.error(f"Error evaluating model: {e}")
        raise

def save_metrics(metrics: Dict[str, float], path: str) -> None:
    """Save evaluation metrics to a JSON file."""
    try:
        with open(path, "w") as f:
            json.dump(metrics, f, indent=4)
        logging.info(f"Metrics saved to {path}")
    except Exception as e:
        logging.error(f"Error saving metrics to {path}: {e}")
        raise

def main() -> None:
    try:
        model = load_model("models/random_forest_model.pkl")
        _, X_test, y_test = load_test_data("data/interim/test_tfidf.csv")
        metrics_dict = evaluate_model(model, X_test, y_test)
        save_metrics(metrics_dict, "reports/metrics.json")
        logging.info("Model evaluation completed successfully.")
    except Exception as e:
        logging.critical(f"Model evaluation failed: {e}")

if __name__ == "__main__":
    main()