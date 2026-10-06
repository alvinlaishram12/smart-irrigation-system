"""
test_edge_cases.py
-------------------
Day 7: systematic edge-case testing. Days 5-6 spot-checked one or two
scenarios manually; this sweeps every crop x growth stage x weather-scenario
combination and reports a pass/fail table, so "tested under simulated
drought and heavy-rainfall scenarios" (the synopsis's own methodology
promise) is backed by an actual, repeatable test run rather than a couple
of manual dashboard clicks.

IMPORTANT METHODOLOGY NOTE (a mistake caught while building this): the first
version of this suite hand-guessed a single "expected" outcome per scenario
and applied it to all 16 crop x growth-stage combinations - e.g. assuming
33% moisture should trigger irrigation for every crop. It shouldn't: Rice's
base threshold (55%) is nearly double Cotton's (28%), and each growth stage
scales that threshold again (Flowering x1.15, Maturity x0.85 - see
simulate_data.py). A flat expectation produced ~40 "failures" that were
really just a wrong test, not a wrong model. Fixed by computing the expected
label programmatically from the exact same threshold formula
simulate_data.py used to generate the training labels in the first place -
this checks "does the trained model reproduce the rule it learned from" and
"do the safety overrides correctly take precedence at the true extremes",
which is what this stage of testing should actually verify.

Usage:
    python src/test_edge_cases.py
"""

import os
import sys

import joblib
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict_live import (build_feature_row, CROPS as CROP_LIST, STAGES, apply_safety_override,  # noqa: E402
                           DEFAULT_MODEL, load_model, predict_probability)
from ood_detector import check_ood, NUMERIC_COLS_FOR_OOD  # noqa: E402
from simulate_data import CROPS as CROP_PROFILES  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Must match simulate_data.py's growth_stage_for_day() sensitivity multipliers exactly.
STAGE_SENSITIVITY = {"Seedling": 0.9, "Vegetative": 1.0, "Flowering": 1.15, "Maturity": 0.85}

# name, weather dict, moisture, category
# category "rule" -> expected outcome computed from the crop/stage threshold rule
# category "safety" -> expected outcome dictated by the hard safety bounds regardless of the rule
SCENARIOS = [
    ("Severe drought", {"temperature_c": 44, "humidity_pct": 14, "rainfall_mm": 0, "rainfall_forecast_next_day_mm": 0},
     18, "safety", True, "Below the 20% safety floor -> must irrigate no matter what"),

    ("Heavy monsoon, saturated soil", {"temperature_c": 27, "humidity_pct": 92, "rainfall_mm": 45, "rainfall_forecast_next_day_mm": 60},
     95, "safety", False, "Above the 92% safety ceiling -> must not irrigate no matter what"),

    ("Dry spell, rain incoming", {"temperature_c": 33, "humidity_pct": 40, "rainfall_mm": 0, "rainfall_forecast_next_day_mm": 25},
     35, "rule", None, "Significant rain forecast (>=5mm) -> rule says skip regardless of moisture"),

    ("Moderate moisture, clear skies", {"temperature_c": 29, "humidity_pct": 45, "rainfall_mm": 0, "rainfall_forecast_next_day_mm": 0},
     33, "rule", None, "No rain relief -> follow the crop/stage moisture threshold exactly"),

    ("Comfortable moisture, mild weather", {"temperature_c": 22, "humidity_pct": 55, "rainfall_mm": 0, "rainfall_forecast_next_day_mm": 0},
     65, "rule", None, "High moisture -> should clear every crop/stage threshold, expect no irrigation"),

    ("Cold winter day, low moisture", {"temperature_c": 4, "humidity_pct": 60, "rainfall_mm": 0, "rainfall_forecast_next_day_mm": 0},
     30, "rule", None, "Temperature doesn't change the moisture rule -> follow the threshold exactly"),

    ("Very low moisture, no rain", {"temperature_c": 38, "humidity_pct": 25, "rainfall_mm": 0, "rainfall_forecast_next_day_mm": 0},
     22, "rule", None, "Below every crop's threshold even at Maturity -> expect irrigation across the board"),
]


def expected_from_rule(crop, stage, moisture, rainfall_forecast):
    """Reproduces simulate_data.py's ground-truth irrigation rule exactly
    (ignoring the 3% random noise flip, which isn't reproducible from a
    static snapshot and isn't something we'd want the model to match anyway)."""
    threshold = CROP_PROFILES[crop]["base_threshold"] * STAGE_SENSITIVITY[stage]
    return moisture < threshold and rainfall_forecast < 5


def run_suite(model_name=DEFAULT_MODEL):
    scaler = joblib.load(f"{MODELS_DIR}/scaler.joblib")
    feature_cols = joblib.load(f"{MODELS_DIR}/feature_columns.joblib")
    model = load_model(model_name, MODELS_DIR)
    thresholds = joblib.load(f"{MODELS_DIR}/tuned_thresholds.joblib")
    threshold = thresholds.get(model_name.lower().replace(" ", "_"), 0.5)
    ood = joblib.load(f"{MODELS_DIR}/ood_detector.joblib")

    rows = []
    for crop in CROP_LIST:
        for stage in STAGES:
            for scenario_name, weather, moisture, category, fixed_expected, rationale in SCENARIOS:
                if category == "safety":
                    expected = fixed_expected
                else:
                    expected = expected_from_rule(crop, stage, moisture, weather["rainfall_forecast_next_day_mm"])

                row = build_feature_row(moisture, crop, stage, weather, feature_cols, scaler)
                prob = predict_probability(model, model_name, row)
                model_says = prob >= threshold
                final_decision, override_reason = apply_safety_override(prob, model_says, moisture)
                ood_result = check_ood(ood["nn"], ood["threshold"], row[NUMERIC_COLS_FOR_OOD].values)

                # For "rule" scenarios, what matters is whether the FINAL decision
                # (model + safety layer) matches the rule; the safety layer is a
                # no-op in the moisture range these particular scenarios use.
                passed = final_decision == expected
                rows.append({
                    "scenario": scenario_name,
                    "category": category,
                    "crop": crop,
                    "stage": stage,
                    "moisture": moisture,
                    "expected": expected,
                    "model_raw": model_says,
                    "final": final_decision,
                    "override_fired": override_reason is not None,
                    "ood_flagged": ood_result["is_ood"],
                    "probability": round(prob, 3),
                    "pass": passed,
                })

    return pd.DataFrame(rows)


def main():
    df = run_suite()
    total = len(df)
    passed = df["pass"].sum()

    print("=" * 70)
    print(f"EDGE-CASE TEST SUITE: {passed}/{total} passed ({passed/total:.1%})")
    print("=" * 70)

    summary = df.groupby("scenario").agg(
        pass_rate=("pass", "mean"),
        override_rate=("override_fired", "mean"),
        ood_rate=("ood_flagged", "mean"),
        n=("pass", "size"),
    ).round(3)
    print("\nBy scenario (across all 16 crop x growth-stage combinations):")
    print(summary.to_string())

    failures = df[~df["pass"]]
    if len(failures):
        print(f"\n{len(failures)} FAILING COMBINATIONS:")
        print(failures[["scenario", "crop", "stage", "moisture", "expected", "final", "probability"]].to_string(index=False))
    else:
        print("\nNo failures - every crop x growth-stage combination behaved as expected in every scenario.")

    out_path = os.path.join(PROJECT_ROOT, "models", "edge_case_test_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nFull results saved to {out_path}")

    return df


if __name__ == "__main__":
    main()

