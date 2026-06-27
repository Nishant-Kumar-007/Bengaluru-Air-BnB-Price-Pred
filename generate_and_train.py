"""
Generate synthetic Bengaluru Airbnb data and train the model.
Run: python3 generate_and_train.py
"""

import os
import sys
import numpy as np
import pandas as pd

np.random.seed(42)
N = 6000

NEIGHBOURHOODS = [
    "A Adugodi","Abbigere","Agara","Arkavathy Layout","Ashokanagar",
    "Banaswadi","Bellandur","BTM Layout","Byatarayanapura","Byrathi",
    "Doresanipalya","Frazer Town","Geddalahalli","Haralur","HBR Layout",
    "Hennur","Hoodi","Horamavu","Hosapalya","HSR Layout","Indiranagar",
    "Jakkur","Jayanagar East","Jayamahal","JP Nagar","Kacharakanahalli",
    "Kalyan Nagar","Kammanahalli","Kasturi Nagar","Kempapura","Kodati",
    "Kodihalli","Kogilu","Koramangala","Kormangala East","Kormangala West",
    "Kothnur","Kundalahalli","Malleswaram","Nagasandra","Nagavara",
    "New Tavarekere","Panathur","Priyadarshini Ward","Puttenahalli",
    "RBI Layout","Richmond Town","Sampigehalli","Shivajinagar","Singasandra",
    "Thanisandra","Vasanth Nagar","Vijaya Bank Layout","Whitefield","Other",
]

ROOM_TYPES = ["Entire home/apt", "Private room", "Shared room", "Hotel room"]

PREMIUM_NEIGHBOURHOODS = {
    "Indiranagar","Koramangala","Whitefield","HSR Layout","Richmond Town",
    "Shivajinagar","Vasanth Nagar","Bellandur","BTM Layout","Malleswaram",
    "Kodihalli","Frazer Town","Jayamahal",
}

neighbourhood   = np.random.choice(NEIGHBOURHOODS, N)
room_type       = np.random.choice(ROOM_TYPES, N, p=[0.55, 0.35, 0.06, 0.04])
accommodates    = np.random.choice([1,2,3,4,5,6,8,10], N, p=[0.10,0.30,0.20,0.18,0.10,0.07,0.03,0.02])
bedrooms        = np.clip(accommodates // 2, 1, 5)
beds            = np.clip(accommodates - np.random.randint(0, 2, N), 1, 10)
bathrooms       = np.round(np.random.choice([1.0,1.5,2.0,2.5,3.0], N, p=[0.50,0.20,0.20,0.06,0.04]), 1)

host_is_superhost      = np.random.choice([0,1], N, p=[0.75,0.25])
host_has_profile_pic   = np.random.choice([0,1], N, p=[0.05,0.95])
host_identity_verified = np.random.choice([0,1], N, p=[0.35,0.65])
host_response_rate     = np.clip(np.random.beta(9, 1, N), 0, 1)
instant_bookable       = np.random.choice([0,1], N, p=[0.55,0.45])

lat_centre, lon_centre = 12.9716, 77.5946
latitude  = np.random.normal(lat_centre,  0.05, N)
longitude = np.random.normal(lon_centre, 0.06, N)

number_of_reviews      = np.random.negative_binomial(3, 0.15, N)
review_scores_rating   = np.clip(np.random.normal(4.4, 0.4, N), 1.0, 5.0).round(1)
review_scores_location = np.clip(np.random.normal(4.3, 0.4, N), 1.0, 5.0).round(1)
minimum_nights         = np.random.choice([1,2,3,5,7,14,30], N, p=[0.45,0.20,0.15,0.08,0.06,0.04,0.02])
availability_365       = np.random.randint(30, 365, N)

# ── Price generation ──────────────────────────────────────────────────────────
# Base nightly rate by neighbourhood tier
base = np.where(np.isin(neighbourhood, list(PREMIUM_NEIGHBOURHOODS)), 5000.0, 2800.0)

# Room type multiplier
room_mult = np.select(
    [room_type == "Entire home/apt", room_type == "Hotel room",
     room_type == "Private room",    room_type == "Shared room"],
    [1.8, 1.6, 1.0, 0.55]
)

# Size multiplier (more guests/rooms → higher price)
size_mult = 1.0 + 0.18 * (accommodates - 2) + 0.05 * (bedrooms - 1)

# Host quality
host_mult = (
    1.0
    + 0.10 * host_is_superhost
    + 0.04 * host_identity_verified
    + 0.06 * host_response_rate
)

# Review quality
review_mult = 1.0 + 0.05 * (review_scores_rating - 4.0) + 0.03 * (review_scores_location - 4.0)

# Minimum nights discount: weekly/monthly stays get lower nightly rate
# 1 night: no discount; 7 nights: ~8%; 14 nights: ~15%; 30 nights: ~25%
nights_discount = np.select(
    [minimum_nights == 1,
     minimum_nights <= 3,
     minimum_nights <= 7,
     minimum_nights <= 14,
     minimum_nights <= 30],
    [1.00, 0.97, 0.92, 0.85, 0.75],
    default=0.70
)

# Availability: low availability (scarce) → slightly higher price
avail_mult = 1.0 + 0.10 * (1.0 - availability_365 / 365.0)

# Instant-bookable slight premium for guests; slight discount for hosts
instant_mult = np.where(instant_bookable == 1, 1.03, 1.00)

# Lognormal noise to simulate real-world price variance
noise = np.random.lognormal(0, 0.22, N)

price = base * room_mult * size_mult * host_mult * review_mult * nights_discount * avail_mult * instant_mult * noise
price = np.clip(price, 500, 35000).round(0)

df = pd.DataFrame({
    "neighbourhood":            neighbourhood,
    "room_type":                room_type,
    "accommodates":             accommodates,
    "bathrooms":                bathrooms,
    "bedrooms":                 bedrooms,
    "beds":                     beds,
    "host_is_superhost":        host_is_superhost,
    "host_has_profile_pic":     host_has_profile_pic,
    "host_identity_verified":   host_identity_verified,
    "host_response_rate":       host_response_rate,
    "instant_bookable":         instant_bookable,
    "latitude":                 latitude,
    "longitude":                longitude,
    "number_of_reviews":        number_of_reviews,
    "review_scores_rating":     review_scores_rating,
    "review_scores_location":   review_scores_location,
    "minimum_nights":           minimum_nights,
    "availability_365":         availability_365,
    "price":                    price,
})

os.makedirs("data", exist_ok=True)
df.to_csv("data/bengaluru_listings.csv", index=False)
print(f"Synthetic dataset saved: {len(df)} rows → data/bengaluru_listings.csv")
print(f"Price range: ₹{df['price'].min():,.0f} – ₹{df['price'].max():,.0f}  (median ₹{df['price'].median():,.0f})")
print("\nAvg price by minimum_nights:")
print(df.groupby("minimum_nights")["price"].mean().round(0).to_string())

from src.Airbnb.pipelines.Training_Pipeline import TrainingPipeline
TrainingPipeline().run()
