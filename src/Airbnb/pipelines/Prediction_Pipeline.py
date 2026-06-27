"""
Prediction Pipeline
====================
Exposes:
  • PredictPipeline   – loads model + preprocessor, runs inference
  • CustomData        – dataclass that mirrors the web form fields and produces
                        a correctly-shaped DataFrame for the pipeline
"""

import os
import sys
import pandas as pd

from src.Airbnb.exception import CustomException
from src.Airbnb.logger import logging
from src.Airbnb.utils import load_object


# ─── Column order must match DataTransformation ──────────────────────────────

_NUMERICAL_FEATURES = [
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

_CATEGORICAL_FEATURES = [
    "room_type",
    "neighbourhood",
    "host_is_superhost",
    "host_has_profile_pic",
    "host_identity_verified",
    "instant_bookable",
]

ALL_FEATURES = _NUMERICAL_FEATURES + _CATEGORICAL_FEATURES


# ─── PredictPipeline ─────────────────────────────────────────────────────────

class PredictPipeline:
    """Loads the saved artifacts and returns a predicted nightly price (INR)."""

    MODEL_PATH       = os.path.join("Artifacts", "model.pkl")
    PREPROCESSOR_PATH = os.path.join("Artifacts", "preprocessor.pkl")

    def predict(self, features: pd.DataFrame) -> list:
        try:
            model        = load_object(self.MODEL_PATH)
            preprocessor = load_object(self.PREPROCESSOR_PATH)

            # Ensure feature order
            features = features[ALL_FEATURES]

            transformed  = preprocessor.transform(features)
            predictions  = model.predict(transformed)

            logging.info(f"Prediction: ₹{predictions[0]:,.0f}")
            return predictions

        except Exception as e:
            raise CustomException(e, sys)


# ─── CustomData ──────────────────────────────────────────────────────────────

class CustomData:
    """
    Holds a single listing's input values and can serialise them to a
    one-row DataFrame ready for the preprocessor.

    Parameters (all match the HTML form field names)
    -------------------------------------------------
    neighbourhood       : e.g. "HSR Layout"
    room_type           : "Entire home/apt" | "Private room" | "Shared room" | "Hotel room"
    accommodates        : number of guests the listing accommodates
    bathrooms           : number of bathrooms (float, e.g. 1.5)
    bedrooms            : number of bedrooms
    beds                : number of beds
    host_is_superhost   : 1 = yes, 0 = no
    host_has_profile_pic: 1 = yes, 0 = no
    host_identity_verified: 1 = yes, 0 = no
    host_response_rate  : 0.0–1.0  (e.g. 0.95 = 95 %)
    instant_bookable    : 1 = yes, 0 = no
    latitude            : decimal degrees
    longitude           : decimal degrees
    number_of_reviews   : total review count
    review_scores_rating: 1.0–5.0
    review_scores_location: 1.0–5.0
    minimum_nights      : minimum stay in nights
    availability_365    : nights available in the next 365 days
    """

    def __init__(
        self,
        neighbourhood:            str,
        room_type:                str,
        accommodates:             int,
        bathrooms:                float,
        bedrooms:                 int,
        beds:                     int,
        host_is_superhost:        int,
        host_has_profile_pic:     int,
        host_identity_verified:   int,
        host_response_rate:       float,
        instant_bookable:         int,
        latitude:                 float,
        longitude:                float,
        number_of_reviews:        int,
        review_scores_rating:     float,
        review_scores_location:   float,
        minimum_nights:           int,
        availability_365:         int,
    ):
        self.neighbourhood             = neighbourhood
        self.room_type                 = room_type
        self.accommodates              = accommodates
        self.bathrooms                 = bathrooms
        self.bedrooms                  = bedrooms
        self.beds                      = beds
        self.host_is_superhost         = host_is_superhost
        self.host_has_profile_pic      = host_has_profile_pic
        self.host_identity_verified    = host_identity_verified
        self.host_response_rate        = host_response_rate
        self.instant_bookable          = instant_bookable
        self.latitude                  = latitude
        self.longitude                 = longitude
        self.number_of_reviews         = number_of_reviews
        self.review_scores_rating      = review_scores_rating
        self.review_scores_location    = review_scores_location
        self.minimum_nights            = minimum_nights
        self.availability_365          = availability_365

    def get_data_as_dataframe(self) -> pd.DataFrame:
        """Return a one-row DataFrame with columns in the order expected by the pipeline."""
        return pd.DataFrame(
            [{
                "accommodates":             self.accommodates,
                "bathrooms":                self.bathrooms,
                "bedrooms":                 self.bedrooms,
                "beds":                     self.beds,
                "host_response_rate":       self.host_response_rate,
                "latitude":                 self.latitude,
                "longitude":                self.longitude,
                "number_of_reviews":        self.number_of_reviews,
                "review_scores_rating":     self.review_scores_rating,
                "review_scores_location":   self.review_scores_location,
                "minimum_nights":           self.minimum_nights,
                "availability_365":         self.availability_365,
                "room_type":                self.room_type,
                "neighbourhood":            self.neighbourhood,
                "host_is_superhost":        self.host_is_superhost,
                "host_has_profile_pic":     self.host_has_profile_pic,
                "host_identity_verified":   self.host_identity_verified,
                "instant_bookable":         self.instant_bookable,
            }]
        )
