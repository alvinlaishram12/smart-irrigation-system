"""
simulate_data.py
-----------------
Generates a realistic, day-by-day simulated smart-irrigation dataset.

Instead of drawing every row independently at random, this simulates an
actual soil-moisture time series per field: moisture drops each day due to
evapotranspiration (driven by temperature & humidity), rises with rainfall,
and is topped up whenever the rule-based irrigation policy fires. This gives
the ML models (trained later) a realistic, temporally-coherent signal to
learn from instead of pure noise.

Usage:
    python src/simulate_data.py --days 730 --seed 42 --out data/irrigation_data.csv
"""

import argparse
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Crop profiles: moisture threshold below which irrigation is triggered,
# and how much a single irrigation event raises soil moisture.
# Thresholds are higher for water-sensitive crops / growth stages.
# ---------------------------------------------------------------------------
CROPS = {
    "Wheat":  {"base_threshold": 32, "irrigation_boost": 14, "season_len": 140},
    "Rice":   {"base_threshold": 55, "irrigation_boost": 16, "season_len": 130},
    "Maize":  {"base_threshold": 35, "irrigation_boost": 13, "season_len": 110},
    "Cotton": {"base_threshold": 28, "irrigation_boost": 12, "season_len": 160},
}

GROWTH_STAGES = ["Seedling", "Vegetative", "Flowering", "Maturity"]


def growth_stage_for_day(day_in_season, season_len):
    """Return growth stage name and a moisture-sensitivity multiplier."""
    frac = (day_in_season % season_len) / season_len
    if frac < 0.20:
        return "Seedling", 0.9
    elif frac < 0.50:
        return "Vegetative", 1.0
    elif frac < 0.75:
        return "Flowering", 1.15  # most water-sensitive stage
    else:
        return "Maturity", 0.85


def seasonal_temperature(day_of_year, noise_rng):
    """Punjab-like seasonal temperature curve (deg C), roughly 5-45."""
    seasonal = 24 + 18 * np.sin(2 * np.pi * (day_of_year - 100) / 365)
    return round(float(seasonal + noise_rng.normal(0, 2.5)), 1)


def seasonal_humidity(day_of_year, rainfall_today, noise_rng):
    """Humidity (%) - higher in monsoon months and on rainy days."""
    seasonal = 55 + 20 * np.sin(2 * np.pi * (day_of_year - 170) / 365)
    rain_bump = min(rainfall_today * 1.5, 20)
    val = seasonal + rain_bump + noise_rng.normal(0, 5)
    return round(float(np.clip(val, 15, 98)), 1)


def seasonal_rainfall(day_of_year, rng):
    """Rainfall (mm) - monsoon-heavy (Jun-Sep, day 152-273), sparse otherwise."""
    is_monsoon = 152 <= day_of_year <= 273
    rain_prob = 0.45 if is_monsoon else 0.10
    if rng.random() < rain_prob:
        return round(float(rng.exponential(18 if is_monsoon else 6)), 1)
    return 0.0


def simulate_field(days, crop_name, start_day_offset, seed):
    """Simulate one field growing one crop over `days` days."""
    rng = np.random.default_rng(seed)
    profile = CROPS[crop_name]
    season_len = profile["season_len"]

    soil_moisture = rng.uniform(45, 60)  # initial condition
    rows = []

    # Pre-compute rainfall for the whole run so we can peek "tomorrow" for
    # the weather-forecast feature (mirrors what the real API integration
    # will provide in Day 5).
    day_of_year_seq = [((start_day_offset + d) % 365) + 1 for d in range(days + 1)]
    rainfall_seq = [seasonal_rainfall(doy, rng) for doy in day_of_year_seq]

    for d in range(days):
        day_of_year = day_of_year_seq[d]
        rainfall_today = rainfall_seq[d]
        rainfall_forecast_tomorrow = rainfall_seq[d + 1]

        temperature = seasonal_temperature(day_of_year, rng)
        humidity = seasonal_humidity(day_of_year, rainfall_today, rng)

        stage, sensitivity = growth_stage_for_day(d, season_len)
        threshold = profile["base_threshold"] * sensitivity

        # Evapotranspiration: hotter & drier air pulls more moisture out.
        et_loss = 2.5 * (0.9 + 0.10 * max(temperature - 20, 0) - 0.04 * (humidity - 50))
        et_loss = max(et_loss, 0.7)

        soil_moisture -= et_loss
        soil_moisture += rainfall_today * 0.6
        soil_moisture = float(np.clip(soil_moisture, 5, 100))

        # --- Rule-based irrigation decision (ground-truth label) ---
        # Irrigate if moisture is below the crop/stage threshold AND no
        # meaningful rain is forecast tomorrow. Small random noise mimics
        # real-world farmer judgement/override.
        would_irrigate = soil_moisture < threshold and rainfall_forecast_tomorrow < 5
        noise_flip = rng.random() < 0.03  # 3% real-world inconsistency
        irrigate = (would_irrigate != noise_flip)

        # IMPORTANT: record the moisture reading *before* any irrigation
        # boost is applied. This is what a real soil sensor would report at
        # the moment the decision is made - and critically, it's what
        # predict_live.py/the dashboard supply at inference time. An earlier
        # version of this script recorded moisture *after* the boost, which
        # silently overwrote genuinely-low pre-decision readings (e.g. 18%)
        # with the boosted post-irrigation value (e.g. 32%) on exactly the
        # rows where irrigate=1 - teaching the model that "irrigate=1" goes
        # with moderate moisture, not low moisture. That bug was only caught
        # in Day 7's systematic edge-case sweep, where querying the model
        # with genuinely low pre-decision moisture (as any real deployment
        # would) produced wrong, underconfident predictions across every
        # crop and growth stage. See README "Day 7" for the full writeup.
        recorded_moisture = round(soil_moisture, 1)

        if irrigate:
            soil_moisture = float(np.clip(soil_moisture + profile["irrigation_boost"], 5, 100))
            water_volume_liters_m2 = round(profile["irrigation_boost"] * 4.2, 1)
        else:
            water_volume_liters_m2 = 0.0

        rows.append({
            "day_index": d,
            "day_of_year": day_of_year,
            "crop_type": crop_name,
            "growth_stage": stage,
            "soil_moisture_pct": recorded_moisture,
            "temperature_c": temperature,
            "humidity_pct": humidity,
            "rainfall_mm": rainfall_today,
            "rainfall_forecast_next_day_mm": rainfall_forecast_tomorrow,
            "irrigate": int(irrigate),
            "water_volume_liters_m2": water_volume_liters_m2,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Simulate smart-irrigation field data.")
    parser.add_argument("--days", type=int, default=730, help="Days to simulate per crop field")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out", type=str, default="data/irrigation_data.csv", help="Output CSV path")
    args = parser.parse_args()

    all_frames = []
    for i, crop in enumerate(CROPS.keys()):
        df = simulate_field(args.days, crop, start_day_offset=i * 37, seed=args.seed + i)
        df.insert(0, "field_id", f"field_{i+1}_{crop.lower()}")
        all_frames.append(df)

    full_df = pd.concat(all_frames, ignore_index=True)
    full_df.to_csv(args.out, index=False)

    print(f"Saved {len(full_df)} rows across {len(CROPS)} simulated fields -> {args.out}")
    print("\nClass balance (irrigate):")
    print(full_df["irrigate"].value_counts(normalize=True).round(3))
    print("\nPer-crop irrigation rate:")
    print(full_df.groupby("crop_type")["irrigate"].mean().round(3))


if __name__ == "__main__":
    main()
