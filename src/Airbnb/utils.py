import os
import sys
import pickle
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.Airbnb.exception import CustomException
from src.Airbnb.logger import logging


def save_object(file_path: str, obj):
    """Persist any Python object to disk as a pickle file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
        logging.info(f"Object saved to {file_path}")
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str):
    """Load a pickled object from disk."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Artifact not found at '{file_path}'. "
                "Please run the training pipeline first."
            )
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models: dict) -> dict:
    """
    Fit each model and return a dict of evaluation metrics.
    Returns {model_name: {'r2': float, 'mae': float, 'rmse': float}}
    """
    report = {}
    try:
        for name, model in models.items():
            logging.info(f"Training {name} ...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            report[name] = {"r2": r2, "mae": mae, "rmse": rmse}
            logging.info(f"{name}  →  R²={r2:.4f}  MAE=₹{mae:.0f}  RMSE=₹{rmse:.0f}")
        return report
    except Exception as e:
        raise CustomException(e, sys)
