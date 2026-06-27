"""
Data Transformation Component
==============================
Builds a sklearn ColumnTransformer that:
  • Imputes + scales all numerical features
  • Imputes + ordinal-encodes all categorical features

The fitted preprocessor is saved to Artifacts/preprocessor.pkl so that
the inference pipeline can apply identical transformations at prediction time.
"""

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.Airbnb.exception import CustomException
from src.Airbnb.logger import logging
from src.Airbnb.utils import save_object


# ─── Config ──────────────────────────────────────────────────────────────────

@dataclass
class DataTransformationConfig:
    preprocessor_path: str = os.path.join("Artifacts", "preprocessor.pkl")


# ─── Feature lists ───────────────────────────────────────────────────────────

NUMERICAL_FEATURES = [
    "accommodates",
    "bathrooms",
    "bedrooms",
    "beds",
    "host_response_rate",
    "latitude",
    "longitude",
    "number_of_reviews",
    "review_scores_rating",
    "review_scores_location",
    "minimum_nights",
    "availability_365",
]

CATEGORICAL_FEATURES = [
    "room_type",
    "neighbourhood",
    "host_is_superhost",
    "host_has_profile_pic",
    "host_identity_verified",
    "instant_bookable",
]

TARGET_COLUMN = "price"


# ─── Main class ──────────────────────────────────────────────────────────────

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    # ── Build preprocessor ───────────────────────────────────────────────────

    def _build_preprocessor(self) -> ColumnTransformer:
        """Return an unfitted ColumnTransformer."""
        numerical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_pipeline,  NUMERICAL_FEATURES),
                ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
            ],
            remainder="drop",
        )
        return preprocessor

    # ── Column guard ─────────────────────────────────────────────────────────

    def _guard_columns(self, df: pd.DataFrame, split_name: str):
        all_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN]
        missing = [c for c in all_features if c not in df.columns]
        if missing:
            logging.warning(
                f"[{split_name}] Missing columns (will be NaN-filled): {missing}"
            )
            for col in missing:
                df[col] = np.nan
        return df

    # ── Public API ───────────────────────────────────────────────────────────

    def initiate_data_transformation(self, train_path: str, test_path: str):
        logging.info("=" * 60)
        logging.info("Data Transformation started")

        try:
            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)

            train_df = self._guard_columns(train_df, "train")
            test_df  = self._guard_columns(test_df,  "test")

            X_train = train_df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].copy()
            y_train = train_df[TARGET_COLUMN].values

            X_test  = test_df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].copy()
            y_test  = test_df[TARGET_COLUMN].values

            preprocessor = self._build_preprocessor()

            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr  = preprocessor.transform(X_test)

            logging.info(
                f"Transformed shapes — X_train: {X_train_arr.shape}, "
                f"X_test: {X_test_arr.shape}"
            )

            save_object(self.config.preprocessor_path, preprocessor)
            logging.info(f"Preprocessor saved → {self.config.preprocessor_path}")
            logging.info("Data Transformation completed")

            return (
                X_train_arr,
                X_test_arr,
                y_train,
                y_test,
                self.config.preprocessor_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
