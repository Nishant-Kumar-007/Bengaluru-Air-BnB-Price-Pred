# 🏠 Bengaluru Airbnb Price Prediction

End-to-end ML pipeline that predicts **nightly Airbnb prices in INR** for
Bengaluru listings, retrained on fresh data from
**[DoorStep Analytics](https://doorstepanalytics.com/report?location=Bengaluru&country=India)**.

> This project is a full rework of the original multi-city Kaggle-dataset repo.
> The old dataset (US cities, USD prices, log-price target) has been replaced by
> a Bengaluru-specific dataset (INR prices, direct price target).

---

## What changed from the original repo

| Aspect | Original (Kaggle) | This version (DoorStep / Bengaluru) |
|--------|-------------------|--------------------------------------|
| Dataset | Multi-city Kaggle CSV (~74k rows) | Bengaluru-only, 9 654 rows, 68 cols |
| Currency | USD | **INR (₹)** |
| Target | `log_price` | `price` (raw INR) |
| `city` feature | NYC, Chicago, LA, DC, SF, Boston | **Removed** (single city) |
| `bed_type` feature | Air bed / Futon / Real Bed… | **Removed** (not in new schema) |
| `cancellation_policy` | strict / moderate / flexible | **Removed** (not in new schema) |
| `cleaning_fee` | separate column | **Removed** (bundled in Airbnb price) |
| New features | — | `neighbourhood`, `minimum_nights`, `availability_365`, `review_scores_location` |
| Form UI | US-centric | **Bengaluru neighbourhoods + ₹ output** |

---

## Project structure

```
.
├── app.py                          ← Flask web app (updated for Bengaluru)
├── requirements.txt
├── setup.py
├── Dockerfile
├── data/
│   └── bengaluru_listings.csv      ← ⬅ YOU PLACE THE DATA HERE (see below)
├── Artifacts/                      ← auto-created during training
│   ├── raw.csv
│   ├── train.csv
│   ├── test.csv
│   ├── preprocessor.pkl
│   └── model.pkl
├── src/Airbnb/
│   ├── exception.py
│   ├── logger.py
│   ├── utils.py
│   ├── components/
│   │   ├── data_ingestion.py       ← reads + cleans bengaluru_listings.csv
│   │   ├── data_transformation.py  ← sklearn ColumnTransformer
│   │   └── model_trainer.py        ← Ridge / RF / GBM / XGB / CatBoost
│   └── pipelines/
│       ├── Training_Pipeline.py    ← chains all three components
│       └── Prediction_Pipeline.py  ← CustomData → DataFrame → prediction
├── templates/
│   ├── index.html                  ← prediction form
│   └── error.html
├── static/css/style.css
└── Notebook_Experiments/
    └── 01_Bengaluru_EDA.py         ← exploratory data analysis
```

---

## Quick start

### 1 — Get the dataset

1. Go to <https://doorstepanalytics.com/report?location=Bengaluru&country=India>
2. Click **Download Data → Airbnb Master Dataset** (9 654 rows · 68 columns · 4.8 MB)
3. Save the file as **`data/bengaluru_listings.csv`** in the project root

> **Alternative (free):** InsideAirbnb publishes Bengaluru data at
> <https://insideairbnb.com/get-the-data/> — download `listings.csv.gz` for
> Bengaluru, decompress it, and save as `data/bengaluru_listings.csv`.
> The column schema is identical; DoorStep Analytics adds extra analytics columns.

### 2 — Install dependencies

```bash
conda create -n airbnb-blr python=3.11 -y
conda activate airbnb-blr
pip install -r requirements.txt
pip install -e .
```

### 3 — Train the model

```bash
python -m src.Airbnb.pipelines.Training_Pipeline
```

This writes three artifacts to `Artifacts/`: `preprocessor.pkl`, `model.pkl`,
and the cleaned CSV splits.

### 4 — Run the web app

```bash
python app.py
```

Open <http://localhost:8080> — fill in listing details, click **Predict Price**,
and receive an estimated nightly rate in INR.

### 5 — (Optional) Docker

```bash
docker build -t airbnb-bengaluru .
docker run -p 8080:8080 airbnb-bengaluru
```

---

## Features used by the model

### Numerical
| Feature | Description |
|---------|-------------|
| `accommodates` | Max number of guests |
| `bathrooms` | Number of bathrooms |
| `bedrooms` | Number of bedrooms |
| `beds` | Number of beds |
| `host_response_rate` | Host response rate (0–1) |
| `latitude` | Listing latitude |
| `longitude` | Listing longitude |
| `number_of_reviews` | Total review count |
| `review_scores_rating` | Overall rating (1–5) |
| `review_scores_location` | Location rating (1–5) |
| `minimum_nights` | Minimum stay |
| `availability_365` | Days available per year |

### Categorical
| Feature | Values |
|---------|--------|
| `room_type` | Entire home/apt · Private room · Shared room · Hotel room |
| `neighbourhood` | 50 + Bengaluru wards (HSR Layout, Koramangala, Whitefield …) |
| `host_is_superhost` | 0 / 1 |
| `host_has_profile_pic` | 0 / 1 |
| `host_identity_verified` | 0 / 1 |
| `instant_bookable` | 0 / 1 |

---

## Models evaluated

| Model | Notes |
|-------|-------|
| Ridge Regression | Linear baseline |
| Random Forest | 200 trees, depth 20 |
| Gradient Boosting | 300 estimators, lr 0.05 |
| XGBoost | 300 estimators, hist method |
| **CatBoost** | **Usually wins on this tabular dataset** |

The best model by R² on the test set is persisted automatically.

---

## Acknowledgements

* Dataset — [DoorStep Analytics](https://doorstepanalytics.com/) /
  [InsideAirbnb](https://insideairbnb.com/)
* Original project structure — [KalyanM45/End-to-End-Airbnb-Price-Prediction](https://github.com/KalyanM45/End-to-End-Airbnb-Price-Prediction)
