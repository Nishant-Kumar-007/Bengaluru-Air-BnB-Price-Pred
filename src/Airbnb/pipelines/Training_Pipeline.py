"""
Training Pipeline
==================
Chains together:
  1. DataIngestion   → reads data/bengaluru_listings.csv, saves Artifacts/train.csv + test.csv
  2. DataTransformation → fits preprocessor, saves Artifacts/preprocessor.pkl
  3. ModelTrainer    → trains models, saves Artifacts/model.pkl

Usage
-----
    python -m src.Airbnb.pipelines.Training_Pipeline
"""

import sys
from src.Airbnb.exception import CustomException
from src.Airbnb.logger import logging

from src.Airbnb.components.data_ingestion import DataIngestion
from src.Airbnb.components.data_transformation import DataTransformation
from src.Airbnb.components.model_trainer import ModelTrainer


class TrainingPipeline:
    def run(self):
        logging.info("╔══════════════════════════════════════════════╗")
        logging.info("║      BENGALURU AIRBNB TRAINING PIPELINE      ║")
        logging.info("╚══════════════════════════════════════════════╝")

        try:
            # ── Step 1: Ingestion ────────────────────────────────────────────
            ingestion = DataIngestion()
            train_path, test_path = ingestion.initiate_data_ingestion()

            # ── Step 2: Transformation ───────────────────────────────────────
            transformation = DataTransformation()
            X_train, X_test, y_train, y_test, _ = \
                transformation.initiate_data_transformation(train_path, test_path)

            # ── Step 3: Training ─────────────────────────────────────────────
            trainer = ModelTrainer()
            best_model_name, metrics = trainer.initiate_model_training(
                X_train, X_test, y_train, y_test
            )

            print("\n✅  Training complete!")
            print(f"   Best model : {best_model_name}")
            print(f"   R²         : {metrics['r2']}")
            print(f"   MAE        : ₹{metrics['mae']:,.2f}")
            print(f"   RMSE       : ₹{metrics['rmse']:,.2f}")

            return best_model_name, metrics

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run()
