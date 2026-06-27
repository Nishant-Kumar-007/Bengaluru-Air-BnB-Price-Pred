"""
Airbnb Price Predictor — Bengaluru
===================================
Flask web app for predicting Airbnb nightly prices (INR) for Bengaluru listings.
Dataset source: https://doorstepanalytics.com/report?location=Bengaluru&country=India
"""

from flask import Flask, request, render_template
from src.Airbnb.pipelines.Prediction_Pipeline import CustomData, PredictPipeline

app = Flask(__name__)

# ─── Bengaluru neighbourhood list (from DoorStep Analytics data) ───────────
NEIGHBOURHOODS = [
    "A Adugodi",
    "Abbigere",
    "Agara",
    "Arkavathy Layout",
    "Ashokanagar",
    "Banaswadi",
    "Bellandur",
    "BTM Layout",
    "Byatarayanapura",
    "Byrathi",
    "Doresanipalya",
    "Frazer Town",
    "Geddalahalli",
    "Haralur",
    "HBR Layout",
    "Hennur",
    "Hoodi",
    "Horamavu",
    "Hosapalya",
    "HSR Layout",
    "Indiranagar",
    "Jakkur",
    "Jayanagar East",
    "Jayamahal",
    "JP Nagar",
    "Kacharakanahalli",
    "Kalyan Nagar",
    "Kammanahalli",
    "Kasturi Nagar",
    "Kempapura",
    "Kodati",
    "Kodihalli",
    "Kogilu",
    "Koramangala",
    "Kormangala East",
    "Kormangala West",
    "Kothnur",
    "Kundalahalli",
    "Malleswaram",
    "Nagasandra",
    "Nagavara",
    "New Tavarekere",
    "Panathur",
    "Priyadarshini Ward",
    "Puttenahalli",
    "RBI Layout",
    "Richmond Town",
    "Sampigehalli",
    "Shivajinagar",
    "Singasandra",
    "Thanisandra",
    "Vasanth Nagar",
    "Vijaya Bank Layout",
    "Whitefield",
    "Other",
]

ROOM_TYPES = [
    "Entire home/apt",
    "Private room",
    "Shared room",
    "Hotel room",
]


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET", "POST"])
def home():
    context = dict(
        neighbourhoods=NEIGHBOURHOODS,
        room_types=ROOM_TYPES,
        final_result=None,
        total_result=None,
        min_nights=None,
        error_message=None,
        form_data={},
    )

    if request.method == "POST":
        try:
            form = request.form

            # Parse host_response_rate: the form sends 0–100, the model wants 0–1
            raw_rate = float(form.get("responserate") or 95)
            response_rate = raw_rate / 100 if raw_rate > 1 else raw_rate

            data = CustomData(
                neighbourhood           = form.get("neighbourhood"),
                room_type               = form.get("roomtype"),
                accommodates            = int(form.get("accommodates") or 2),
                bathrooms               = float(form.get("bathrooms") or 1),
                bedrooms                = int(form.get("bedrooms") or 1),
                beds                    = int(form.get("beds") or 1),
                host_is_superhost       = int(form.get("superhost") or 0),
                host_has_profile_pic    = int(form.get("profilepic") or 1),
                host_identity_verified  = int(form.get("verified") or 0),
                host_response_rate      = response_rate,
                instant_bookable        = int(form.get("instantbook") or 0),
                latitude                = float(form.get("lat") or 12.9716),
                longitude               = float(form.get("long") or 77.5946),
                number_of_reviews       = int(form.get("reviews") or 0),
                review_scores_rating    = float(form.get("rating") or 4.5),
                review_scores_location  = float(form.get("locationrating") or 4.5),
                minimum_nights          = int(form.get("minnights") or 1),
                availability_365        = int(form.get("availability") or 180),
            )

            df = data.get_data_as_dataframe()
            pipeline = PredictPipeline()
            pred = pipeline.predict(df)

            nightly = round(pred[0])
            min_n   = int(form.get("minnights") or 1)
            context["final_result"] = f"₹{nightly:,}"
            context["total_result"] = f"₹{nightly * min_n:,}"
            context["min_nights"]   = min_n
            context["form_data"]    = form.to_dict()

        except Exception as e:
            context["error_message"] = f"Prediction error: {str(e)}"

    return render_template("index.html", **context)


@app.route("/health")
def health():
    return {"status": "ok"}, 200


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
