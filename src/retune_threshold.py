"""
retune_threshold.py
--------------------
Day 7 finding: the systematic edge-case sweep (test_edge_cases.py) showed
the Random Forest failing 100% of the "very low moisture, no rain" scenario
across all 16 crop x growth-stage combinations - moisture at 22%, clearly
below every crop's threshold, with no rain relief, yet the model predicted
"do not irrigate" with only ~20% probability. None of these were flagged as
out-of-distribution, so this isn't the Day 5/6 extrapolation problem; it's
the model itself under-predicting the positive class.

This traces back to a choice made in Day 3: the decision threshold was
tuned to maximise F1, which weighs precision and recall equally. But in
irrigation, the two error types are NOT equally costly: a false negative
(failing to irrigate a thirsty field) risks real crop damage, while a false
positive (irrigating a field that didn't strictly need it yet) wastes some
water but causes no lasting harm. That asymmetry means the threshold should
be biased toward recall, not sit at the F1-optimal point.

This script re-tunes the threshold using F-beta with beta=2 (recall
weighted 2x as heavily as precision), using the exact same validation fold
as Day 3's tune_threshold() - never touching the test set - so this is a
principled re-calibration, not test-set fitting.

Usage:
    python src/retune_threshold.py --data data/processed --out models --beta 2.0
"""

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, precision_recall_curve, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit


def tune_threshold_fbeta(model, X_train, y_train, tscv, beta=2.0):
    """Same validation split Day 3 used for F1 tuning, optimizing F-beta instead."""
    train_idx, val_idx = list(tscv.split(X_train))[-1]
    X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
    val_proba = model.predict_proba(X_val)[:, 1]

    precisions, recalls, thresholds = precision_recall_curve(y_val, val_proba)
    beta_sq = beta ** 2
    fbetas = (1 + beta_sq) * precisions * recalls / (beta_sq * precisions + recalls + 1e-9)
    best_idx = np.argmax(fbetas[:-1])
    return float(thresholds[best_idx]) if len(thresholds) else 0.5


def main():
    parser = argparse.ArgumentParser(description="Re-tune thresholds toward recall using F-beta.")
    parser.add_argument("--data", type=str, default="data/processed")
    parser.add_argument("--out", type=str, default="models")
    parser.add_argument("--beta", type=float, default=2.0,
                         help="Recall weight relative to precision (2.0 = recall matters 2x as much)")
    args = parser.parse_args()

    X_train = pd.read_csv(f"{args.data}/X_train.csv")
    y_train = pd.read_csv(f"{args.data}/y_train_irrigate.csv").iloc[:, 0]
    X_test = pd.read_csv(f"{args.data}/X_test.csv")
    y_test = pd.read_csv(f"{args.data}/y_test_irrigate.csv").iloc[:, 0]

    tscv = TimeSeriesSplit(n_splits=5)

    old_thresholds = joblib.load(f"{args.out}/tuned_thresholds.joblib")
    new_thresholds = {}

    print(f"{'Model':<16} {'Old thr.':>9} {'New thr.':>9} | {'Old P/R/F1':>16} | {'New P/R/F1':>16}")
    print("-" * 80)

    for key, filename in [("decision_tree", "decision_tree.joblib"), ("random_forest", "random_forest.joblib")]:
        model = joblib.load(f"{args.out}/{filename}")
        new_threshold = tune_threshold_fbeta(model, X_train, y_train, tscv, beta=args.beta)
        new_thresholds[key] = new_threshold

        test_proba = model.predict_proba(X_test)[:, 1]

        old_pred = (test_proba >= old_thresholds[key]).astype(int)
        new_pred = (test_proba >= new_threshold).astype(int)

        old_stats = (precision_score(y_test, old_pred, zero_division=0),
                     recall_score(y_test, old_pred, zero_division=0),
                     f1_score(y_test, old_pred, zero_division=0))
        new_stats = (precision_score(y_test, new_pred, zero_division=0),
                     recall_score(y_test, new_pred, zero_division=0),
                     f1_score(y_test, new_pred, zero_division=0))

        print(f"{key:<16} {old_thresholds[key]:>9.3f} {new_threshold:>9.3f} | "
              f"{old_stats[0]:.2f}/{old_stats[1]:.2f}/{old_stats[2]:.2f}     | "
              f"{new_stats[0]:.2f}/{new_stats[1]:.2f}/{new_stats[2]:.2f}")

    joblib.dump(new_thresholds, f"{args.out}/tuned_thresholds.joblib")
    print(f"\nSaved recall-weighted (F{args.beta:.0f}-optimal) thresholds to {args.out}/tuned_thresholds.joblib")
    print("(P/R/F1 = Precision/Recall/F1 on the held-out TEST set, shown only to illustrate the")
    print(" trade-off - the threshold itself was chosen on the validation fold, not the test set)")


if __name__ == "__main__":
    main()
