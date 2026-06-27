"""
Bengaluru Airbnb — Exploratory Data Analysis
=============================================
Run this as a script  OR  open as a Jupyter notebook (jupytext compatible).

Dataset: data/bengaluru_listings.csv
Source : https://doorstepanalytics.com/report?location=Bengaluru&country=India
"""

# %% ── 0. Imports ─────────────────────────────────────────────────────────────
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams["figure.dpi"] = 120

DATA_PATH = "../data/bengaluru_listings.csv"

# %% ── 1. Load raw data ───────────────────────────────────────────────────────
raw = pd.read_csv(DATA_PATH, low_memory=False)
print(f"Shape: {raw.shape}")
raw.head(3)

# %% ── 2. Column overview ─────────────────────────────────────────────────────
print(raw.dtypes.to_string())
print("\nNull %:")
null_pct = (raw.isnull().sum() / len(raw) * 100).sort_values(ascending=False)
print(null_pct[null_pct > 0].to_string())

# %% ── 3. Price cleaning ──────────────────────────────────────────────────────
def clean_price(v):
    if pd.isna(v):
        return np.nan
    txt = re.sub(r"[Rs₹$,\s]", "", str(v))
    try:
        return float(txt)
    except ValueError:
        return np.nan

raw["price_clean"] = raw["price"].apply(clean_price)

# Remove extreme outliers for visualisation
prices = raw["price_clean"].dropna()
prices_iqr = prices[(prices >= 300) & (prices <= 30_000)]
print(f"Price stats (300–30k INR):\n{prices_iqr.describe().round(0)}")

# %% ── 4. Price distribution ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(prices_iqr, bins=60, color="#FF5A5F", edgecolor="white", linewidth=0.4)
axes[0].set_xlabel("Nightly Price (INR)")
axes[0].set_title("Price Distribution")

axes[1].hist(np.log1p(prices_iqr), bins=60, color="#484848", edgecolor="white", linewidth=0.4)
axes[1].set_xlabel("log(1 + Price)")
axes[1].set_title("Log-Price Distribution")

plt.tight_layout()
plt.savefig("price_distribution.png", bbox_inches="tight")
plt.show()

# %% ── 5. Room type breakdown ─────────────────────────────────────────────────
if "room_type" in raw.columns:
    rt_price = (
        raw[["room_type", "price_clean"]]
        .dropna()
        .groupby("room_type")["price_clean"]
        .median()
        .sort_values(ascending=False)
    )
    rt_price.plot(kind="bar", color="#FF5A5F", figsize=(8, 4))
    plt.title("Median Nightly Price by Room Type")
    plt.ylabel("INR")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig("price_by_room_type.png", bbox_inches="tight")
    plt.show()

# %% ── 6. Top-20 neighbourhoods ───────────────────────────────────────────────
nbh_col = "neighbourhood_cleansed" if "neighbourhood_cleansed" in raw.columns else "neighbourhood"

if nbh_col in raw.columns:
    top20 = (
        raw.groupby(nbh_col)["price_clean"]
        .agg(["median", "count"])
        .query("count >= 15")
        .sort_values("median", ascending=True)
        .tail(20)
    )

    top20["median"].plot(kind="barh", color="#FF5A5F", figsize=(9, 7))
    plt.title("Top-20 Neighbourhoods — Median Nightly Price (INR)")
    plt.xlabel("Median Price (INR)")
    plt.tight_layout()
    plt.savefig("price_by_neighbourhood.png", bbox_inches="tight")
    plt.show()

# %% ── 7. Bedrooms vs. price ──────────────────────────────────────────────────
if "bedrooms" in raw.columns:
    bed_price = (
        raw[["bedrooms", "price_clean"]]
        .dropna()
        .query("bedrooms <= 7")
        .groupby("bedrooms")["price_clean"]
        .median()
    )
    bed_price.plot(kind="bar", color="#767676", figsize=(8, 4))
    plt.title("Median Price by Bedrooms")
    plt.ylabel("INR")
    plt.xlabel("Bedrooms")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("price_by_bedrooms.png", bbox_inches="tight")
    plt.show()

# %% ── 8. Numeric correlation heatmap ────────────────────────────────────────
num_cols = [
    "price_clean", "accommodates", "bedrooms", "beds",
    "number_of_reviews", "review_scores_rating",
    "minimum_nights", "availability_365",
]
available = [c for c in num_cols if c in raw.columns]
corr = raw[available].corr()

plt.figure(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap="RdYlGn", center=0, linewidths=0.5)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", bbox_inches="tight")
plt.show()

# %% ── 9. Superhost price premium ─────────────────────────────────────────────
if "host_is_superhost" in raw.columns:
    sh = raw[["host_is_superhost", "price_clean"]].dropna()
    sh["host_is_superhost"] = sh["host_is_superhost"].map({"t": "Superhost", "f": "Regular"})
    sns.boxplot(data=sh.query("price_clean < 15000"),
                x="host_is_superhost", y="price_clean",
                palette={"Superhost": "#FF5A5F", "Regular": "#767676"})
    plt.title("Price Distribution: Superhost vs Regular")
    plt.ylabel("Nightly Price (INR)")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig("superhost_price.png", bbox_inches="tight")
    plt.show()

# %% ── 10. Geographic scatter ────────────────────────────────────────────────
if {"latitude", "longitude", "price_clean"}.issubset(raw.columns):
    geo = raw[["latitude", "longitude", "price_clean"]].dropna()
    geo = geo.query("price_clean < 20000")

    plt.figure(figsize=(9, 7))
    sc = plt.scatter(geo["longitude"], geo["latitude"],
                     c=geo["price_clean"], cmap="RdYlGn_r",
                     alpha=0.35, s=6)
    plt.colorbar(sc, label="Nightly Price (INR)")
    plt.title("Bengaluru Airbnb Listings — Price Heat Map")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.savefig("geo_price_scatter.png", bbox_inches="tight")
    plt.show()

print("\n✅  EDA complete. Charts saved in Notebook_Experiments/")
