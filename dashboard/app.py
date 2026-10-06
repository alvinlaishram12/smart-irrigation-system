"""
app.py — Smart Irrigation Dashboard
------------------------------------
Wraps Day 5's predict_live.py logic in an interactive Streamlit UI:
- Sensor + crop inputs in the sidebar
- Live weather panel (with the Day 5 offline-fallback + OOD warnings surfaced,
  not hidden)
- Irrigation recommendation with probability
- Historical soil-moisture trend for the selected crop's simulated field
- Model comparison table from Day 3/4

Run:
    streamlit run dashboard/app.py
"""

import os
import sys
from datetime import date

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Make src/ importable regardless of where streamlit is launched from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

from weather_api import fetch_weather                       # noqa: E402
from ood_detector import check_ood, NUMERIC_COLS_FOR_OOD     # noqa: E402
from predict_live import (build_feature_row, CROPS, STAGES, MODEL_FILES, SKLEARN_MODELS,  # noqa: E402
                           DEFAULT_MODEL, load_model, predict_probability, apply_safety_override)

st.set_page_config(page_title="Smart Irrigation Dashboard", page_icon="🌱", layout="wide")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "irrigation_data.csv")


# ---------------------------------------------------------------------------
# Cached loaders — models/data load once per session, not on every rerun.
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    return {
        "scaler": joblib.load(f"{MODELS_DIR}/scaler.joblib"),
        "feature_cols": joblib.load(f"{MODELS_DIR}/feature_columns.joblib"),
        "thresholds": joblib.load(f"{MODELS_DIR}/tuned_thresholds.joblib"),
        "ood": joblib.load(f"{MODELS_DIR}/ood_detector.joblib"),
        "models": {name: load_model(name, MODELS_DIR) for name in MODEL_FILES},
    }


@st.cache_data
def load_historical_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data(ttl=600)
def cached_weather(lat, lon):
    return fetch_weather(lat, lon)


artifacts = load_artifacts()
history = load_historical_data()

# ---------------------------------------------------------------------------
# Sidebar — inputs
# ---------------------------------------------------------------------------
st.sidebar.header("🌾 Field Conditions")
crop_type = st.sidebar.selectbox("Crop type", CROPS, index=CROPS.index("Wheat"))
growth_stage = st.sidebar.selectbox("Growth stage", STAGES, index=STAGES.index("Vegetative"))
soil_moisture = st.sidebar.slider("Current soil moisture (%)", 5, 100, 40)
model_choice = st.sidebar.selectbox("Model", list(MODEL_FILES.keys()), index=list(MODEL_FILES.keys()).index(DEFAULT_MODEL))

st.sidebar.header("📍 Location")
lat = st.sidebar.number_input("Latitude", value=30.9010, format="%.4f")
lon = st.sidebar.number_input("Longitude", value=75.8573, format="%.4f")

st.sidebar.header("🌦️ Weather Source")
use_live = st.sidebar.toggle("Use live weather API", value=True,
                              help="Turn off to manually enter weather for testing edge cases (e.g. drought, heavy rain).")

if use_live:
    weather = cached_weather(lat, lon)
else:
    st.sidebar.caption("Manual weather entry:")
    weather = {
        "temperature_c": st.sidebar.slider("Temperature (°C)", -5, 50, 30),
        "humidity_pct": st.sidebar.slider("Humidity (%)", 10, 100, 50),
        "rainfall_mm": st.sidebar.slider("Rainfall today (mm)", 0, 100, 0),
        "rainfall_forecast_next_day_mm": st.sidebar.slider("Rainfall forecast tomorrow (mm)", 0, 100, 0),
        "is_live": False,
        "source": "Manual entry",
    }

# ---------------------------------------------------------------------------
# Main panel — header
# ---------------------------------------------------------------------------
st.title("🌱 AI-Based Smart Irrigation Dashboard")
st.caption("Software-simulation smart irrigation system — Days 1–7 complete")

# ---------------------------------------------------------------------------
# Weather panel
# ---------------------------------------------------------------------------
st.subheader("Current Conditions")
if not weather.get("is_live", True):
    if use_live:
        st.warning("⚠️ Live weather request failed — showing offline sample data. "
                   "Check your internet connection before relying on this for a real decision.")
    else:
        st.info("ℹ️ Manual weather entry mode — adjust sliders in the sidebar to test edge cases.")

wcol1, wcol2, wcol3, wcol4 = st.columns(4)
wcol1.metric("Temperature", f"{weather['temperature_c']} °C")
wcol2.metric("Humidity", f"{weather['humidity_pct']} %")
wcol3.metric("Rainfall today", f"{weather['rainfall_mm']} mm")
wcol4.metric("Rain forecast tomorrow", f"{weather['rainfall_forecast_next_day_mm']} mm")

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
model = artifacts["models"][model_choice]
threshold = artifacts["thresholds"].get(model_choice.lower().replace(" ", "_"), 0.5)

row = build_feature_row(soil_moisture, crop_type, growth_stage, weather,
                         artifacts["feature_cols"], artifacts["scaler"])
probability = predict_probability(model, model_choice, row)
model_says_irrigate = probability >= threshold
irrigate, override_reason = apply_safety_override(probability, model_says_irrigate, soil_moisture)
ood_result = check_ood(artifacts["ood"]["nn"], artifacts["ood"]["threshold"],
                        row[NUMERIC_COLS_FOR_OOD].values)

st.subheader("Recommendation")

rec_col, prob_col = st.columns([2, 1])
with rec_col:
    if irrigate:
        st.success(f"### 💧 IRRIGATE TODAY\n{crop_type} ({growth_stage}) at {soil_moisture}% soil moisture")
    else:
        st.info(f"### ✅ NO IRRIGATION NEEDED\n{crop_type} ({growth_stage}) at {soil_moisture}% soil moisture")
    if override_reason:
        st.error(f"🛑 **Safety override active** — {override_reason}")
    if ood_result["is_ood"]:
        st.warning(
            f"⚠️ **Unusual input combination** (distance {ood_result['distance']} > "
            f"threshold {ood_result['threshold']}). This sensor + weather combination "
            "doesn't closely match anything in the training data — the model is "
            "extrapolating, so treat this prediction with extra caution."
        )
with prob_col:
    st.metric("Irrigation probability", f"{probability:.1%}")
    st.caption(f"Decision threshold: {threshold:.1%} (model: {model_choice})")

st.divider()

# ---------------------------------------------------------------------------
# Historical trend for the selected crop
# ---------------------------------------------------------------------------
st.subheader(f"Historical Soil Moisture — {crop_type} Field")
field_data = history[history["crop_type"] == crop_type].copy()
field_id = field_data["field_id"].iloc[0]
field_data = field_data[field_data["field_id"] == field_id].sort_values("day_index")

window = st.slider("Days to show", 30, min(365, len(field_data)), 120)
plot_data = field_data.tail(window).set_index("day_index")

chart_data = plot_data[["soil_moisture_pct"]].rename(columns={"soil_moisture_pct": "Soil Moisture (%)"})
st.line_chart(chart_data, height=300)

irrigation_days = plot_data[plot_data["irrigate"] == 1]
st.caption(f"{len(irrigation_days)} irrigation events in the last {window} simulated days for this field "
           f"({len(irrigation_days) / window:.1%} of days).")

st.divider()

# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------
st.subheader("Model Comparison (Days 3–4 results)")
comparison_path = f"{MODELS_DIR}/final_model_comparison.csv"
if os.path.exists(comparison_path):
    comparison_df = pd.read_csv(comparison_path)
    st.dataframe(comparison_df, width="stretch", hide_index=True)
    st.caption("Random Forest is used as the default model above based on this comparison "
               "(highest F1-score and ROC-AUC) — see README for full discussion.")
else:
    st.info("Run `train_baseline_models.py` and `train_neural_network.py` to populate this table.")

st.divider()
st.caption(
    "⚠️ This is a software-simulation project (B.Tech CSE synopsis). Soil moisture and crop data are "
    "simulated, not from physical sensors. Weather is live via Open-Meteo when available."
)
