"""
Data Ingestion Component
========================
Loads the Bengaluru Airbnb listings CSV downloaded from:
  https://doorstepanalytics.com/report?location=Bengaluru&country=India

The CSV is expected at:  data/bengaluru_listings.csv   (configurable via DATA_SOURCE_PATH)

Column naming follows the standard InsideAirbnb / DoorStep Analytics schema
(68-column Master Dataset).  The component selects the features we need, cleans
them, and produces train / test splits under Artifacts/.
"""

import os
import re
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.Airbnb.exception import CustomException
from src.Airbnb.logger import logging


# ─── Config ──────────────────────────────────────────────────────────────────

@dataclass
class DataIngestionConfig:
    source_data_path: str = os.path.join("data", "bengaluru_listings.csv")
    raw_data_path:    str = os.path.join("Artifacts", "raw.csv")
    train_data_path:  str = os.path.join("Artifacts", "train.csv")
    test_data_path:   str = os.path.join("Artifacts", "test.csv")

    # Price bounds in INR — outliers beyond these are dropped
    price_min: float = 300.0
    price_max: float = 150_000.0

    test_size:    float = 0.20
    random_state: int   = 42


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _clean_price(value) -> float:
    """Strip currency symbols (Rs, ₹, $) and commas, return float."""
    if pd.isna(value):
        return np.nan
    txt = re.sub(r"[Rs₹$,\s]", "", str(value))
    try:
        return float(txt)
    except ValueError:
        return np.nan


def _clean_bathrooms(text) -> float:
    """Extract the leading numeric value from strings like '1 bath', '1.5 baths', 'Shared half-bath'."""
    if pd.isna(text):
        return np.nan
    m = re.search(r"[\d.]+", str(text))
    return float(m.group()) if m else np.nan


def _clean_response_rate(value) -> float:
    """Convert '95%' → 0.95, or a raw 0-100 number → fraction."""
    if pd.isna(value):
        return np.nan
    txt = str(value).replace("%", "").strip()
    try:
        v = float(txt)
        return v / 100 if v > 1 else v
    except ValueError:
        return np.nan


def _bool_flag(value) -> float:
    """Map Airbnb 't'/'f' flags (or True/False) to 1/0."""
    if pd.isna(value):
        return np.nan
    mapping = {"t": 1, "f": 0, "true": 1, "false": 0, True: 1, False: 0}
    return mapping.get(str(value).lower().strip(), np.nan)


# ─── Main class ──────────────────────────────────────────────────────────────

class DataIngestion:
    """
    Reads the raw Bengaluru CSV, selects + cleans the relevant columns,
    removes outliers, and splits into train / test sets.
    """

    # Maps DoorStep / InsideAirbnb column names → our internal names.
    # We try the primary name first; if absent we fall back to the alias list.
    COLUMN_ALIASES = {
        "neighbourhood":         ["neighbourhood_cleansed", "neighbourhood"],
        "room_type":             ["room_type"],
        "property_type":         ["property_type"],
        "accommodates":          ["accommodates"],
        "bathrooms":             ["bathrooms_text", "bathrooms"],
        "bedrooms":              ["bedrooms"],
        "beds":                  ["beds"],
        "host_is_superhost":     ["host_is_superhost"],
        "host_has_profile_pic":  ["host_has_profile_pic"],
        "host_identity_verified":["host_identity_verified"],
        "host_response_rate":    ["host_response_rate"],
        "instant_bookable":      ["instant_bookable"],
        "latitude":              ["latitude"],
        "longitude":             ["longitude"],
        "number_of_reviews":     ["number_of_reviews"],
        "review_scores_rating":  ["review_scores_rating"],
        "review_scores_location":["review_scores_location"],
        "minimum_nights":        ["minimum_nights"],
        "availability_365":      ["availability_365"],
        "price":                 ["price"],
    }

    def __init__(self):
        self.config = DataIngestionConfig()

    # ── Private helpers ──────────────────────────────────────────────────────

    def _resolve_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select columns using the alias map; warn when a column is missing."""
        selected = {}
        for internal_name, candidates in self.COLUMN_ALIASES.items():
            for c in candidates:
                if c in df.columns:
                    selected[c] = internal_name
                    break
            else:
                logging.warning(f"Column '{internal_name}' not found in dataset — will be NaN.")
        return df.rename(columns=selected)[list(selected.values())]

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply type coercions and cleaning rules."""
        # Price
        if "price" in df.columns:
            df["price"] = df["price"].apply(_clean_price)

        # Bathrooms (may be a text column in newer InsideAirbnb data)
        if "bathrooms" in df.columns and df["bathrooms"].dtype == object:
            df["bathrooms"] = df["bathrooms"].apply(_clean_bathrooms)

        # Host response rate
        if "host_response_rate" in df.columns:
            df["host_response_rate"] = df["host_response_rate"].apply(_clean_response_rate)

        # Boolean flags  (t/f strings from Airbnb API)
        bool_cols = [
            "host_is_superhost", "host_has_profile_pic",
            "host_identity_verified", "instant_bookable",
        ]
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].apply(_bool_flag)

        # Numeric casts
        num_cols = [
            "accommodates", "bedrooms", "beds",
            "number_of_reviews", "minimum_nights", "availability_365",
            "latitude", "longitude",
            "review_scores_rating", "review_scores_location",
        ]
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.dropna(subset=["price"])
        df = df[
            (df["price"] >= self.config.price_min) &
            (df["price"] <= self.config.price_max)
        ]
        logging.info(
            f"Outlier removal: {before} → {len(df)} rows "
            f"(kept prices ₹{self.config.price_min:,.0f}–₹{self.config.price_max:,.0f})"
        )
        return df

    # ── Public API ───────────────────────────────────────────────────────────

    def initiate_data_ingestion(self):
        logging.info("=" * 60)
        logging.info("Data Ingestion started")

        try:
            if not os.path.exists(self.config.source_data_path):
                raise FileNotFoundError(
                    f"Source data not found at '{self.config.source_data_path}'.\n"
                    "Please download the Bengaluru Master Dataset from:\n"
                    "  https://doorstepanalytics.com/report?location=Bengaluru&country=India\n"
                    "and place it at: data/bengaluru_listings.csv"
                )

            raw = pd.read_csv(self.config.source_data_path, low_memory=False)
            logging.info(f"Loaded raw CSV: {raw.shape[0]:,} rows × {raw.shape[1]} columns")

            df = self._resolve_columns(raw)
            df = self._clean(df)
            df = self._remove_outliers(df)
            df = df.reset_index(drop=True)

            logging.info(f"Final dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

            # Save raw (cleaned) snapshot
            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)
            df.to_csv(self.config.raw_data_path, index=False)

            # Train / test split
            train, test = train_test_split(
                df,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
            )
            train.to_csv(self.config.train_data_path, index=False)
            test.to_csv(self.config.test_data_path,  index=False)

            logging.info(
                f"Train: {len(train):,} rows | Test: {len(test):,} rows  "
                f"→ saved to Artifacts/"
            )
            logging.info("Data Ingestion completed")

            return self.config.train_data_path, self.config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)
