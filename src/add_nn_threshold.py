"""
add_nn_threshold.py
--------------------
Day 8 finding: train_neural_network.py (Day 4) computed an F1-optimal
threshold (0.72) for the Neural Network but never persisted it anywhere.
predict_live.py and test_edge_cases.py both look it up as
tuned_thresholds["neural_network"], which never existed - so every live
prediction and the entire Day 7 edge-case suite silently used the sklearn
default of 0.5 instead. This went unnoticed because 0.5 is a *plausible*
threshold, not a crash - exactly the kind of quiet correctness bug that
survives until someone checks the actual values in the saved artifact
instead of trusting the code path.

This script closes the gap two ways:
1. Adds the missing "neural_network" key so the lookup Day 7's code already
   expects actually resolves.
2. While fixing it, applies the SAME recall-weighted F-beta(beta=2)
   methodology Day 7 used for Decision Tree/Random Forest, rather than
   just hard-coding Day 4's plain-F1-optimal 0.72 - for consistency, all
   three deployed models should be threshold-tuned by the same principle
   (a missed irrigation costs more than an extra one).

Uses the pure-NumPy inference path (nn_numpy_inference.py), not the Keras
model directly - consistent with Day 8's broader goal of keeping
TensorFlow out of anything that runs as part of the live system.

Usage:
    python src/add_nn_threshold.py --data data/processed --out models --beta 2.0
"""

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit

from nn_numpy_inference import load_numpy_model, predict_proba_numpy


def tune_threshold_fbeta_numpy(weights, X_train, y_train, tscv, beta=2.0):
    """Mirrors retune_threshold.py's tune_threshold_fbeta(), but scores the
    NumPy-reimplemented network instead of calling an sklearn .predict_proba."""
    train_idx, val_idx = list(tscv.split(X_train))[-1]
    X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
    val_proba = predict_proba_numpy(X_val.values, weights)

    precisions, recalls, thresholds = precision_recall_curve(y_val, val_proba)
    beta_sq = beta ** 2
    fbetas = (1 + beta_sq) * precisions * recalls / (beta_sq * precisions + recalls + 1e-9)
    best_idx = np.argmax(fbetas[:-1])
    return float(thresholds[best_idx]) if len(thresholds) else 0.5


def main():
    parser = argparse.ArgumentParser(description="Add the missing Neural Network threshold.")
    parser.add_argument("--data", type=str, default="data/processed")
    parser.add_argument("--out", type=str, default="models")
    parser.add_argument("--beta", type=float, default=2.0)
    args = parser.parse_args()

    X_train = pd.read_csv(f"{args.data}/X_train.csv")
    y_train = pd.read_csv(f"{args.data}/y_train_irrigate.csv").iloc[:, 0]
    X_test = pd.read_csv(f"{args.data}/X_test.csv")
    y_test = pd.read_csv(f"{args.data}/y_test_irrigate.csv").iloc[:, 0]

    weights = load_numpy_model(f"{args.out}/neural_network_weights.npz")
    tscv = TimeSeriesSplit(n_splits=5)

    new_threshold = tune_threshold_fbeta_numpy(weights, X_train, y_train, tscv, beta=args.beta)

    test_proba = predict_proba_numpy(X_test.values, weights)
    default_pred = (test_proba >= 0.5).astype(int)
    new_pred = (test_proba >= new_threshold).astype(int)

    default_stats = (precision_score(y_test, default_pred, zero_division=0),
                      recall_score(y_test, default_pred, zero_division=0),
                      f1_score(y_test, default_pred, zero_division=0))
    new_stats = (precision_score(y_test, new_pred, zero_division=0),
                 recall_score(y_test, new_pred, zero_division=0),
                 f1_score(y_test, new_pred, zero_division=0))

    thresholds = joblib.load(f"{args.out}/tuned_thresholds.joblib")
    print(f"Existing thresholds (untouched): {thresholds}")
    thresholds["neural_network"] = new_threshold
    joblib.dump(thresholds, f"{args.out}/tuned_thresholds.joblib")

    print(f"\n{'Model':<16} {'Old thr.':>9} {'New thr.':>9} | {'Old P/R/F1':>16} | {'New P/R/F1':>16}")
    print("-" * 80)
    print(f"{'neural_network':<16} {0.5:>9.3f} {new_threshold:>9.3f} | "
          f"{default_stats[0]:.2f}/{default_stats[1]:.2f}/{default_stats[2]:.2f}     | "
          f"{new_stats[0]:.2f}/{new_stats[1]:.2f}/{new_stats[2]:.2f}")
    print(f"\nSaved. tuned_thresholds.joblib now has keys: {list(thresholds.keys())}")


if __name__ == "__main__":
    main()
