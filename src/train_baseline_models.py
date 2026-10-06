"""
train_baseline_models.py
-------------------------
Trains and evaluates two baseline classifiers for the irrigate/do-not-irrigate
decision: Decision Tree and Random Forest. Also reports a "majority class"
dummy baseline for context - a model is only useful if it clearly beats
always predicting the most common class.

Handles class imbalance (~17% positive) via class_weight="balanced" rather
than resampling, so the test-set evaluation stays on the real distribution.

Usage:
    python src/train_baseline_models.py --data data/processed --out models
"""

import argparse
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score, precision_score,
                              recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier


def load_data(data_dir):
    X_train = pd.read_csv(f"{data_dir}/X_train.csv")
    X_test = pd.read_csv(f"{data_dir}/X_test.csv")
    y_train = pd.read_csv(f"{data_dir}/y_train_irrigate.csv").squeeze()
    y_test = pd.read_csv(f"{data_dir}/y_test_irrigate.csv").squeeze()
    return X_train, X_test, y_train, y_test


def evaluate(name, model, X_test, y_test, results, figures_dir):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "model": name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
    }
    if y_proba is not None and len(set(y_test)) > 1:
        metrics["roc_auc"] = round(roc_auc_score(y_test, y_proba), 4)

    results.append(metrics)

    print(f"\n{'='*50}\n{name}\n{'='*50}")
    print(classification_report(y_test, y_pred, target_names=["No Irrigation", "Irrigate"], zero_division=0))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4.5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix — {name}")
    plt.xticks([0, 1], ["No Irrigation", "Irrigate"])
    plt.yticks([0, 1], ["No Irrigation", "Irrigate"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                      color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    plt.colorbar(fraction=0.046)
    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_")
    plt.savefig(f"{figures_dir}/confusion_matrix_{safe_name}.png", dpi=150)
    plt.close()

    return y_proba


def plot_feature_importance(name, model, feature_names, figures_dir):
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    plt.figure(figsize=(8, 6))
    importances.plot.barh(color="#2E5395")
    plt.title(f"Feature Importance — {name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_")
    plt.savefig(f"{figures_dir}/feature_importance_{safe_name}.png", dpi=150)
    plt.close()
    return importances


def plot_roc_curves(curves, figures_dir):
    plt.figure(figsize=(6, 6))
    for name, (y_test, y_proba) in curves.items():
        if y_proba is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Model Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/roc_comparison.png", dpi=150)
    plt.close()


def tune_threshold(model, X_train, y_train, tscv):
    """Find the F1-optimal decision threshold using the last CV fold as a
    validation set (never touches the test set, so evaluation stays honest)."""
    from sklearn.metrics import precision_recall_curve

    train_idx, val_idx = list(tscv.split(X_train))[-1]
    X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
    val_proba = model.predict_proba(X_val)[:, 1]

    precisions, recalls, thresholds = precision_recall_curve(y_val, val_proba)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1s[:-1])  # last point has no corresponding threshold
    return float(thresholds[best_idx]) if len(thresholds) else 0.5


def evaluate_at_threshold(name, y_test, y_proba, threshold, results):
    y_pred = (y_proba >= threshold).astype(int)
    metrics = {
        "model": f"{name} (tuned threshold={threshold:.2f})",
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        # ROC-AUC is threshold-invariant (it's computed over all thresholds),
        # so it's identical to the base model's AUC - included here too so
        # every row in the comparison table has a consistent set of keys.
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4) if len(set(y_test)) > 1 else None,
    }
    results.append(metrics)
    print(f"\n{'='*50}\n{metrics['model']}\n{'='*50}")
    print(classification_report(y_test, y_pred, target_names=["No Irrigation", "Irrigate"], zero_division=0))
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train baseline irrigation classifiers.")
    parser.add_argument("--data", type=str, default="data/processed")
    parser.add_argument("--out", type=str, default="models")
    parser.add_argument("--figures", type=str, default="notebooks/figures")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.figures, exist_ok=True)

    X_train, X_test, y_train, y_test = load_data(args.data)
    feature_names = X_train.columns.tolist()
    results = []
    roc_curves = {}

    # ---- Baseline: always predict the majority class ----
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    evaluate("Majority Class Baseline", dummy, X_test, y_test, results, args.figures)

    # ---- Decision Tree (small grid search over depth, time-series CV) ----
    tscv = TimeSeriesSplit(n_splits=4)
    dt_grid = GridSearchCV(
        DecisionTreeClassifier(class_weight="balanced", random_state=42),
        param_grid={"max_depth": [3, 5, 7, 10, None], "min_samples_leaf": [1, 5, 10]},
        scoring="f1", cv=tscv, n_jobs=-1,
    )
    dt_grid.fit(X_train, y_train)
    dt_best = dt_grid.best_estimator_
    print(f"\nBest Decision Tree params: {dt_grid.best_params_}")
    dt_proba = evaluate("Decision Tree", dt_best, X_test, y_test, results, args.figures)
    roc_curves["Decision Tree"] = (y_test, dt_proba)
    plot_feature_importance("Decision Tree", dt_best, feature_names, args.figures)
    joblib.dump(dt_best, f"{args.out}/decision_tree.joblib")
    dt_threshold = tune_threshold(dt_best, X_train, y_train, tscv)
    evaluate_at_threshold("Decision Tree", y_test, dt_proba, dt_threshold, results)

    # ---- Random Forest (small grid search) ----
    rf_grid = GridSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
        param_grid={"n_estimators": [100, 200], "max_depth": [7, 10, None], "min_samples_leaf": [1, 5]},
        scoring="f1", cv=tscv, n_jobs=-1,
    )
    rf_grid.fit(X_train, y_train)
    rf_best = rf_grid.best_estimator_
    print(f"\nBest Random Forest params: {rf_grid.best_params_}")
    rf_proba = evaluate("Random Forest", rf_best, X_test, y_test, results, args.figures)
    roc_curves["Random Forest"] = (y_test, rf_proba)
    plot_feature_importance("Random Forest", rf_best, feature_names, args.figures)
    joblib.dump(rf_best, f"{args.out}/random_forest.joblib")
    rf_threshold = tune_threshold(rf_best, X_train, y_train, tscv)
    evaluate_at_threshold("Random Forest", y_test, rf_proba, rf_threshold, results)

    joblib.dump({"decision_tree": dt_threshold, "random_forest": rf_threshold},
                f"{args.out}/tuned_thresholds.joblib")

    plot_roc_curves(roc_curves, args.figures)

    # ---- Comparison table ----
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{args.out}/baseline_model_comparison.csv", index=False)
    with open(f"{args.out}/baseline_model_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*50}\nMODEL COMPARISON SUMMARY\n{'='*50}")
    print(results_df.to_string(index=False))

    # Comparison bar chart
    plt.figure(figsize=(9, 5))
    metrics_to_plot = ["accuracy", "precision", "recall", "f1_score"]
    x = np.arange(len(metrics_to_plot))
    width = 0.25
    for i, row in results_df.iterrows():
        plt.bar(x + i * width, [row[m] for m in metrics_to_plot], width, label=row["model"])
    plt.xticks(x + width, metrics_to_plot)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Baseline Model Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{args.figures}/baseline_model_comparison.png", dpi=150)
    plt.close()

    print(f"\nSaved models to {args.out}/, figures to {args.figures}/")


if __name__ == "__main__":
    main()
