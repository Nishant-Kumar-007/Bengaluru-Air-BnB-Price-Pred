"""
Model Trainer Component
========================
Trains a suite of regression models on the Bengaluru Airbnb data,
selects the best by R² on the held-out test set, and persists it.

Models evaluated
-----------------
• Ridge Regression   (linear baseline)
• Random Forest
• Gradient Boosting
• XGBoost
• CatBoost           (typically best on tabular data with categoricals)
"""

import os
import sys
import numpy as np
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.Airbnb.exception import CustomException
from src.Airbnb.logger import logging
from src.Airbnb.utils import evaluate_models, save_object


# ─── Config ──────────────────────────────────────────────────────────────────

@dataclass
class ModelTrainerConfig:
    model_path:  str   = os.path.join("Artifacts", "model.pkl")
    min_r2:      float = 0.50   # minimum acceptable R² — raises if not met


# ─── Main class ──────────────────────────────────────────────────────────────

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_training(
        self,
        X_train: np.ndarray,
        X_test:  np.ndarray,
        y_train: np.ndarray,
        y_test:  np.ndarray,
    ):
        logging.info("=" * 60)
        logging.info("Model Training started")

        try:
            models = {
                "Ridge": Ridge(alpha=50.0),
                "RandomForest": RandomForestRegressor(
                    n_estimators=150,
                    max_depth=10,
                    min_samples_leaf=8,
                    max_features=0.6,
                    random_state=42,
                    n_jobs=-1,
                ),
                "GradientBoosting": GradientBoostingRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.75,
                    min_samples_leaf=10,
                    random_state=42,
                ),
                "XGBoost": XGBRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.75,
                    colsample_bytree=0.7,
                    reg_alpha=1.0,
                    reg_lambda=5.0,
                    min_child_weight=5,
                    random_state=42,
                    verbosity=0,
                    tree_method="hist",
                ),
                "CatBoost": CatBoostRegressor(
                    iterations=300,
                    learning_rate=0.05,
                    depth=5,
                    l2_leaf_reg=10,
                    min_data_in_leaf=10,
                    random_seed=42,
                    verbose=0,
                ),
            }

            results = evaluate_models(X_train, y_train, X_test, y_test, models)

            # ── Summary table ────────────────────────────────────────────────
            logging.info("\n{'Model':<22} {'R²':>8} {'MAE (₹)':>12} {'RMSE (₹)':>12}")
            logging.info("-" * 56)
            for name, m in sorted(results.items(), key=lambda x: -x[1]["r2"]):
                logging.info(
                    f"{name:<22} {m['r2']:>8.4f} {m['mae']:>12,.0f} {m['rmse']:>12,.0f}"
                )

            best_name = max(results, key=lambda k: results[k]["r2"])
            best_r2   = results[best_name]["r2"]

            if best_r2 < self.config.min_r2:
                raise ValueError(
                    f"Best model R²={best_r2:.4f} is below threshold "
                    f"{self.config.min_r2}. "
                    "Check data quality or tune hyperparameters."
                )

            # Re-fit best model on full training data before saving
            best_model = models[best_name]
            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)

            final_metrics = {
                "r2":   round(r2_score(y_test, y_pred), 4),
                "mae":  round(mean_absolute_error(y_test, y_pred), 2),
                "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
            }

            logging.info(
                f"\nBest model → {best_name}\n"
                f"  R²   = {final_metrics['r2']}\n"
                f"  MAE  = ₹{final_metrics['mae']:,.2f}\n"
                f"  RMSE = ₹{final_metrics['rmse']:,.2f}"
            )

            save_object(self.config.model_path, best_model)
            logging.info(f"Model saved → {self.config.model_path}")
            logging.info("Model Training completed")

            return best_name, final_metrics

        except Exception as e:
            raise CustomException(e, sys)
