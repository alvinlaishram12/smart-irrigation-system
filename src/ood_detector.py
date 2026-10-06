"""
ood_detector.py
----------------
Fits a simple nearest-neighbor out-of-distribution (OOD) detector on the
training features. Motivation (found during Day 5 integration testing):

When soil moisture (from a real sensor) and humidity (from a live weather
API) are supplied independently at inference time, their combination can
fall outside anything the model saw during training - even though each
value individually looks perfectly reasonable. Soil moisture and humidity
are correlated 0.64 in the training data (see Day 2 EDA); a query with low
moisture but unusually high humidity sits in a sparse corner of feature
space where a Random Forest's predictions are essentially extrapolation
guesses, not learned behaviour. This produced a genuinely non-monotonic,
untrustworthy probability curve when tested (see README "Day 5" notes).

This module doesn't fix the underlying extrapolation - no classifier
truly "fixes" that - but it detects it, so predict_live.py / the Day 6
dashboard can surface a clear warning instead of silently presenting an
unreliable probability as if it were as trustworthy as an in-distribution
prediction.

Usage:
    python src/ood_detector.py --data data/processed --out models
"""

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

NUMERIC_COLS_FOR_OOD = [
    "soil_moisture_pct", "temperature_c", "humidity_pct",
    "rainfall_mm", "rainfall_forecast_next_day_mm",
]


def fit_ood_detector(X_train, percentile=97):
    """
    Fits k-NN on the numeric training features and calibrates a distance
    threshold as the `percentile`-th percentile of each training point's
    own leave-one-out distance to its nearest neighbour. A new point whose
    nearest-neighbour distance exceeds this threshold is flagged as OOD.
    """
    X_numeric = X_train[NUMERIC_COLS_FOR_OOD].values
    nn = NearestNeighbors(n_neighbors=2).fit(X_numeric)  # k=2: self + nearest other
    distances, _ = nn.kneighbors(X_numeric)
    self_distances = distances[:, 1]  # column 0 is distance to self (=0)
    threshold = float(np.percentile(self_distances, percentile))
    return nn, threshold


def check_ood(nn, threshold, query_row_scaled):
    """query_row_scaled: 1-row DataFrame/array with the same scaled numeric
    columns (in NUMERIC_COLS_FOR_OOD order) used to fit the detector."""
    distance, _ = nn.kneighbors(np.asarray(query_row_scaled).reshape(1, -1), n_neighbors=1)
    dist = float(distance[0, 0])
    return {
        "is_ood": dist > threshold,
        "distance": round(dist, 3),
        "threshold": round(threshold, 3),
    }


def main():
    parser = argparse.ArgumentParser(description="Fit the OOD detector on training data.")
    parser.add_argument("--data", type=str, default="data/processed")
    parser.add_argument("--out", type=str, default="models")
    parser.add_argument("--percentile", type=float, default=97)
    args = parser.parse_args()

    X_train = pd.read_csv(f"{args.data}/X_train.csv")
    nn, threshold = fit_ood_detector(X_train, percentile=args.percentile)

    joblib.dump({"nn": nn, "threshold": threshold, "columns": NUMERIC_COLS_FOR_OOD},
                f"{args.out}/ood_detector.joblib")

    # Quick self-check: what fraction of the TEST set (unseen days) gets flagged?
    X_test = pd.read_csv(f"{args.data}/X_test.csv")
    X_test_numeric = X_test[NUMERIC_COLS_FOR_OOD].values
    distances, _ = nn.kneighbors(X_test_numeric, n_neighbors=1)
    flagged_frac = (distances[:, 0] > threshold).mean()

    print(f"OOD detector fit on {len(X_train)} training points.")
    print(f"Distance threshold (percentile={args.percentile}): {threshold:.3f}")
    print(f"Fraction of held-out TEST set flagged as OOD: {flagged_frac:.1%} "
          f"(expected to be low - test set IS in-distribution, just future days)")
    print(f"Saved to {args.out}/ood_detector.joblib")


if __name__ == "__main__":
    main()
