"""
preprocess.py
-------------
Turns the raw simulated dataset into model-ready train/test sets.

Key design decisions (worth explaining in your viva):

1. LEAKAGE: `water_volume_liters_m2` is deterministically derived from the
   `irrigate` label (0 when not irrigated, a fixed crop-specific amount when
   irrigated) - correlation 0.99. It is dropped from the classification
   feature set entirely and kept only as a separate regression target.

2. TIME-BASED SPLIT, NOT RANDOM: this is a time series per field. A random
   80/20 shuffle would let the model "see the future" (e.g. train on day 400,
   test on day 200 of the same field), which inflates accuracy in a way that
   would not hold up in real deployment. Instead, for each field, the first
   80% of days become train and the last 20% become test - mimicking how
   the model would actually be used (trained on history, evaluated on what
   comes next).

3. CYCLICAL DATE ENCODING: day_of_year is encoded as sin/cos pairs instead
   of a raw 1-365 integer, so the model understands that day 365 and day 1
   are adjacent (December-to-January wraparound), not far apart.

Usage:
    python src/preprocess.py --data data/irrigation_data.csv --out data/processed
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


CLASSIFICATION_FEATURES = [
    "soil_moisture_pct", "temperature_c", "humidity_pct",
    "rainfall_mm", "rainfall_forecast_next_day_mm",
    "day_of_year_sin", "day_of_year_cos",
]
CATEGORICAL_FEATURES = ["crop_type", "growth_stage"]
CLASSIFICATION_TARGET = "irrigate"
REGRESSION_TARGET = "water_volume_liters_m2"


def add_cyclical_date_features(df):
    df = df.copy()
    df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
    return df


def time_based_split(df, test_frac=0.2):
    """Split each field's time series so the LAST test_frac of days is test."""
    train_parts, test_parts = [], []
    for field_id, group in df.groupby("field_id"):
        group = group.sort_values("day_index")
        cutoff = int(len(group) * (1 - test_frac))
        train_parts.append(group.iloc[:cutoff])
        test_parts.append(group.iloc[cutoff:])
    return pd.concat(train_parts).reset_index(drop=True), pd.concat(test_parts).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Preprocess the irrigation dataset.")
    parser.add_argument("--data", type=str, default="data/irrigation_data.csv")
    parser.add_argument("--out", type=str, default="data/processed")
    parser.add_argument("--test_frac", type=float, default=0.2)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    df = pd.read_csv(args.data)
    df = add_cyclical_date_features(df)

    # One-hot encode categoricals (fit categories on the full dataset so
    # train/test share identical columns; this doesn't leak label info).
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, prefix=CATEGORICAL_FEATURES)
    encoded_cat_cols = [c for c in df_encoded.columns
                        if any(c.startswith(p + "_") for p in CATEGORICAL_FEATURES)]
    # pandas one-hot columns are bool; cast to int so every downstream
    # consumer (Keras included) sees plain numeric dtypes, not object arrays.
    df_encoded[encoded_cat_cols] = df_encoded[encoded_cat_cols].astype(int)

    feature_cols = CLASSIFICATION_FEATURES + encoded_cat_cols

    train_df, test_df = time_based_split(df_encoded, test_frac=args.test_frac)

    X_train, X_test = train_df[feature_cols].copy(), test_df[feature_cols].copy()
    y_train_clf, y_test_clf = train_df[CLASSIFICATION_TARGET], test_df[CLASSIFICATION_TARGET]
    y_train_reg, y_test_reg = train_df[REGRESSION_TARGET], test_df[REGRESSION_TARGET]

    # Scale only the true numeric columns; leave one-hot columns as 0/1.
    numeric_cols = CLASSIFICATION_FEATURES
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

    # Persist everything Day 3 will need.
    X_train.to_csv(f"{args.out}/X_train.csv", index=False)
    X_test.to_csv(f"{args.out}/X_test.csv", index=False)
    y_train_clf.to_csv(f"{args.out}/y_train_irrigate.csv", index=False)
    y_test_clf.to_csv(f"{args.out}/y_test_irrigate.csv", index=False)
    y_train_reg.to_csv(f"{args.out}/y_train_water_volume.csv", index=False)
    y_test_reg.to_csv(f"{args.out}/y_test_water_volume.csv", index=False)

    joblib.dump(scaler, "models/scaler.joblib")
    joblib.dump(feature_cols, "models/feature_columns.joblib")

    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"\nTrain class balance:\n{y_train_clf.value_counts(normalize=True).round(3)}")
    print(f"\nTest class balance:\n{y_test_clf.value_counts(normalize=True).round(3)}")
    print(f"\nSaved processed splits + scaler to {args.out}/ and models/")


if __name__ == "__main__":
    main()
