"""
predict_live.py
----------------
Ties everything together: takes today's sensor readings (soil moisture,
crop, growth stage) plus a live weather forecast (Day 5), feeds them through
the Day 2 preprocessing pipeline and the Day 3/4 best model (Random Forest),
and prints an irrigation recommendation.

This is the same prediction path the Day 6 Streamlit dashboard will call -
built here first as a plain CLI so the logic can be verified independently
of any UI.

Usage:
    python src/predict_live.py --soil-moisture 38 --crop Wheat --stage Flowering \
        --lat 30.9010 --lon 75.8573
"""

import argparse
import os
from datetime import date

import joblib
import numpy as np
import pandas as pd

from weather_api import fetch_weather
from ood_detector import check_ood, NUMERIC_COLS_FOR_OOD
from nn_numpy_inference import load_numpy_model, predict_proba_numpy

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

CROPS = ["Cotton", "Maize", "Rice", "Wheat"]
STAGES = ["Flowering", "Maturity", "Seedling", "Vegetative"]

# Day 7 update: after fixing the Day 1 moisture-recording bug (see README),
# the Neural Network turned out to be the best model (F1=0.72 vs Random
# Forest's 0.50) - reversing Day 4's conclusion, which was drawn on leaky
# data. predict_live.py originally only supported the two scikit-learn
# models (loaded via joblib); Keras models load and predict differently, so
# that's handled explicitly below rather than pretending one code path
# covers both.
MODEL_FILES = {
    "Decision Tree": "decision_tree.joblib",
    "Random Forest": "random_forest.joblib",
    "Neural Network": "neural_network_weights.npz",
}
SKLEARN_MODELS = {"Decision Tree", "Random Forest"}
DEFAULT_MODEL = "Neural Network"

# Day 8: the Neural Network is served via a pure-NumPy reimplementation of
# its forward pass (see nn_numpy_inference.py), not the original Keras
# model. Importing TensorFlow alone measured ~700MB RAM - enough on its own
# to risk crashing a Streamlit Community Cloud free-tier deployment (1GB
# limit) before the app does anything. The NumPy version is verified
# numerically identical to the trained Keras model (max diff ~4.5e-07, zero
# decision changes on the held-out test set) and removes TensorFlow from
# the live/deployed code path entirely. TensorFlow remains a dev-only
# dependency for training (train_neural_network.py), never imported here.


def load_model(model_name, models_dir=DEFAULT_MODELS_DIR):
    path = f"{models_dir}/{MODEL_FILES[model_name]}"
    if model_name in SKLEARN_MODELS:
        return joblib.load(path)
    return load_numpy_model(path)


def predict_probability(model, model_name, row):
    """Returns P(irrigate) as a plain float, regardless of model type."""
    if model_name in SKLEARN_MODELS:
        return float(model.predict_proba(row)[0, 1])
    proba = predict_proba_numpy(row.values, model)  # model is the weights dict here
    return float(proba[0])

# --- Safety-override thresholds ---------------------------------------
# Found during Day 6 dashboard testing: a true drought scenario (15%
# moisture, 45C, 15% humidity, no rain) got predicted as "do not irrigate"
# (P=0.21, threshold=0.465). Root cause: zero training rows ever combine
# moisture<20% with humidity<25% - the simulation's seasonal humidity model
# never produces hot+bone-dry+parched-soil all at once as often as reality
# might. The model isn't "wrong" so much as unsupported by any training
# evidence in that corner of feature space, and a Random Forest's leaves
# default to whatever's most common nearby, which is unsafe here.
#
# Rather than trust an unsupported extrapolation on a safety-relevant
# decision, hard agronomic bounds override the model outside the range it
# was actually trained on. This is standard practice for ML in
# safety-relevant control loops: the model handles the nuanced middle
# ground; hard bounds handle the extremes.
CRITICAL_LOW_MOISTURE = 20   # below this, irrigate regardless of model output
CRITICAL_HIGH_MOISTURE = 92  # above this, never irrigate regardless of model output


def apply_safety_override(probability, model_says_irrigate, soil_moisture_pct):
    """Returns (final_decision, override_reason_or_None)."""
    if soil_moisture_pct < CRITICAL_LOW_MOISTURE:
        return True, (f"Safety override: soil moisture {soil_moisture_pct}% is below the "
                       f"critical floor ({CRITICAL_LOW_MOISTURE}%) the model was ever trained on - "
                       "irrigating regardless of model output to avoid crop damage.")
    if soil_moisture_pct > CRITICAL_HIGH_MOISTURE:
        return False, (f"Safety override: soil moisture {soil_moisture_pct}% is above the "
                        f"critical ceiling ({CRITICAL_HIGH_MOISTURE}%) - withholding irrigation "
                        "regardless of model output to avoid waterlogging/runoff.")
    return model_says_irrigate, None


def build_feature_row(soil_moisture_pct, crop_type, growth_stage, weather, feature_cols, scaler):
    """Builds a single-row DataFrame matching the exact column order/scaling
    the models were trained on (Day 2's preprocess.py)."""
    today = date.today()
    day_of_year = today.timetuple().tm_yday

    raw = {
        "soil_moisture_pct": soil_moisture_pct,
        "temperature_c": weather["temperature_c"],
        "humidity_pct": weather["humidity_pct"],
        "rainfall_mm": weather["rainfall_mm"],
        "rainfall_forecast_next_day_mm": weather["rainfall_forecast_next_day_mm"],
        "day_of_year_sin": np.sin(2 * np.pi * day_of_year / 365),
        "day_of_year_cos": np.cos(2 * np.pi * day_of_year / 365),
    }
    for c in CROPS:
        raw[f"crop_type_{c}"] = 1 if crop_type == c else 0
    for s in STAGES:
        raw[f"growth_stage_{s}"] = 1 if growth_stage == s else 0

    row = pd.DataFrame([raw])[feature_cols]  # enforce exact training column order

    numeric_cols = ["soil_moisture_pct", "temperature_c", "humidity_pct",
                     "rainfall_mm", "rainfall_forecast_next_day_mm",
                     "day_of_year_sin", "day_of_year_cos"]
    row[numeric_cols] = scaler.transform(row[numeric_cols])
    return row


def recommend(soil_moisture_pct, crop_type, growth_stage, lat, lon,
              model_name=DEFAULT_MODEL, models_dir=DEFAULT_MODELS_DIR):
    if crop_type not in CROPS:
        raise ValueError(f"crop_type must be one of {CROPS}")
    if growth_stage not in STAGES:
        raise ValueError(f"growth_stage must be one of {STAGES}")

    scaler = joblib.load(f"{models_dir}/scaler.joblib")
    feature_cols = joblib.load(f"{models_dir}/feature_columns.joblib")
    model = load_model(model_name, models_dir)
    thresholds = joblib.load(f"{models_dir}/tuned_thresholds.joblib")
    threshold = thresholds.get(model_name.lower().replace(" ", "_"), 0.5)
    ood = joblib.load(f"{models_dir}/ood_detector.joblib")

    weather = fetch_weather(lat, lon)
    row = build_feature_row(soil_moisture_pct, crop_type, growth_stage, weather, feature_cols, scaler)

    probability = predict_probability(model, model_name, row)
    model_says_irrigate = probability >= threshold
    final_irrigate, override_reason = apply_safety_override(probability, model_says_irrigate, soil_moisture_pct)
    ood_result = check_ood(ood["nn"], ood["threshold"], row[NUMERIC_COLS_FOR_OOD].values)

    return {
        "irrigate": bool(final_irrigate),
        "model_says_irrigate": bool(model_says_irrigate),
        "safety_override_reason": override_reason,
        "probability": round(probability, 3),
        "threshold_used": round(threshold, 3),
        "model": model_name,
        "weather": weather,
        "ood": ood_result,
        "inputs": {
            "soil_moisture_pct": soil_moisture_pct,
            "crop_type": crop_type,
            "growth_stage": growth_stage,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Get a live irrigation recommendation.")
    parser.add_argument("--soil-moisture", type=float, required=True, help="Current soil moisture (%)")
    parser.add_argument("--crop", type=str, required=True, choices=CROPS)
    parser.add_argument("--stage", type=str, required=True, choices=STAGES)
    parser.add_argument("--lat", type=float, default=30.9010, help="Latitude (default: Ludhiana, Punjab)")
    parser.add_argument("--lon", type=float, default=75.8573, help="Longitude (default: Ludhiana, Punjab)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, choices=list(MODEL_FILES.keys()))
    args = parser.parse_args()

    result = recommend(args.soil_moisture, args.crop, args.stage, args.lat, args.lon, args.model)

    print("\n" + "=" * 55)
    print("  SMART IRRIGATION RECOMMENDATION")
    print("=" * 55)
    if not result["weather"]["is_live"]:
        print("  ⚠️  WARNING: using OFFLINE sample weather data")
        print("     (live Open-Meteo request failed - check internet)")
    if result["ood"]["is_ood"]:
        print("  ⚠️  WARNING: this sensor+weather combination is UNUSUAL")
        print(f"     (distance {result['ood']['distance']} > threshold {result['ood']['threshold']})")
        print("     Prediction may be unreliable - model is extrapolating")
        print("     beyond combinations seen during training.")
    print(f"  Crop / Stage:        {result['inputs']['crop_type']} / {result['inputs']['growth_stage']}")
    print(f"  Soil moisture:       {result['inputs']['soil_moisture_pct']}%")
    print(f"  Temperature today:   {result['weather']['temperature_c']} C")
    print(f"  Humidity today:      {result['weather']['humidity_pct']}%")
    print(f"  Rainfall today:      {result['weather']['rainfall_mm']} mm")
    print(f"  Rainfall forecast (tomorrow): {result['weather']['rainfall_forecast_next_day_mm']} mm")
    print("-" * 55)
    print(f"  Model used:          {result['model']}")
    print(f"  Irrigation probability: {result['probability']:.1%}")
    print(f"  Decision threshold:  {result['threshold_used']:.1%}")
    if result["safety_override_reason"]:
        print(f"  🛑 {result['safety_override_reason']}")
    decision = "IRRIGATE TODAY" if result["irrigate"] else "DO NOT IRRIGATE"
    print(f"  >>> RECOMMENDATION: {decision} <<<")
    print("=" * 55)


if __name__ == "__main__":
    main()
