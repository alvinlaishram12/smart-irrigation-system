"""
train_neural_network.py
------------------------
Trains a feedforward neural network (Keras) for the irrigate/do-not-irrigate
decision, then produces the FINAL three-way comparison against the Day 3
Decision Tree and Random Forest baselines.

Design notes:

1. VALIDATION SPLIT: a further time-based split carves the last 20% of the
   *training* set off as a validation set for early stopping. The test set
   from Day 2 is never touched until final evaluation, so early-stopping
   decisions can't leak test information.

2. CLASS IMBALANCE: handled via Keras' `class_weight` argument (same
   philosophy as `class_weight="balanced"` in Day 3's sklearn models) rather
   than oversampling, so the loss function - not the data - compensates for
   the ~17% positive rate.

3. ARCHITECTURE: kept deliberately small (2 hidden layers, dropout) because
   the dataset is only ~2,300 training rows - a large network would overfit
   long before it out-learns a Random Forest on tabular data this size.

Usage:
    python src/train_neural_network.py --data data/processed --out models --figures notebooks/figures
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
import tensorflow as tf
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score, precision_score,
                              recall_score, roc_auc_score, roc_curve)

tf.random.set_seed(42)
# tf.random.set_seed alone does NOT fully fix results run-to-run: it leaves
# numpy's and Python's own RNGs (used internally by Keras for batch
# shuffling) unseeded, and CPU op-level parallelism can still introduce
# nondeterminism independent of any seed. Found the hard way in Day 7: a
# fresh run of this exact script, on the exact same data, produced a
# meaningfully different result (F1 0.53 vs an earlier run's 0.44) large
# enough to flip which model looked best overall. tf.keras.utils.set_random_seed
# seeds Python/NumPy/TF together in one call, and enable_op_determinism
# forces deterministic kernels, closing both gaps.
tf.keras.utils.set_random_seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass  # older TF versions may not support this; seeding above still helps substantially


def load_data(data_dir):
    X_train = pd.read_csv(f"{data_dir}/X_train.csv")
    X_test = pd.read_csv(f"{data_dir}/X_test.csv")
    y_train = pd.read_csv(f"{data_dir}/y_train_irrigate.csv").squeeze()
    y_test = pd.read_csv(f"{data_dir}/y_test_irrigate.csv").squeeze()
    return X_train, X_test, y_train, y_test


def time_based_val_split(X_train, y_train, val_frac=0.2):
    """Carve the LAST val_frac rows off as validation (already time-ordered
    per field from Day 2's preprocessing, and fields are contiguous blocks)."""
    cutoff = int(len(X_train) * (1 - val_frac))
    return (X_train.iloc[:cutoff], y_train.iloc[:cutoff],
            X_train.iloc[cutoff:], y_train.iloc[cutoff:])


def build_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    return model


def evaluate_and_plot(name, y_test, y_proba, threshold, figures_dir):
    y_pred = (y_proba >= threshold).astype(int)
    metrics = {
        "model": name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }
    print(f"\n{'='*50}\n{name}\n{'='*50}")
    print(classification_report(y_test, y_pred, target_names=["No Irrigation", "Irrigate"], zero_division=0))

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
    plt.savefig(f"{figures_dir}/confusion_matrix_neural_network.png", dpi=150)
    plt.close()

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train a neural network irrigation classifier.")
    parser.add_argument("--data", type=str, default="data/processed")
    parser.add_argument("--out", type=str, default="models")
    parser.add_argument("--figures", type=str, default="notebooks/figures")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.figures, exist_ok=True)

    X_train_full, X_test, y_train_full, y_test = load_data(args.data)
    X_train, y_train, X_val, y_val = time_based_val_split(X_train_full, y_train_full)

    # class_weight dict, same "balanced" logic as sklearn's class_weight="balanced"
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    class_weight = {0: len(y_train) / (2 * neg), 1: len(y_train) / (2 * pos)}
    print(f"Class weights: {class_weight}")

    model = build_model(X_train.shape[1])
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=15, restore_best_weights=True)

    history = model.fit(
        X_train.values, y_train.values,
        validation_data=(X_val.values, y_val.values),
        epochs=args.epochs, batch_size=32,
        class_weight=class_weight,
        callbacks=[early_stop],
        verbose=2,
    )

    # --- Training curves ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.history["loss"], label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["auc"], label="Train AUC")
    axes[1].plot(history.history["val_auc"], label="Val AUC")
    axes[1].set_title("ROC-AUC over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(f"{args.figures}/neural_network_training_curves.png", dpi=150)
    plt.close()

    # --- Find F1-optimal threshold on the validation set only ---
    val_proba = model.predict(X_val.values, verbose=0).ravel()
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_proba)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_threshold = float(thresholds[np.argmax(f1s[:-1])]) if len(thresholds) else 0.5
    print(f"\nF1-optimal threshold (from validation set): {best_threshold:.3f}")

    # --- Final test-set evaluation ---
    test_proba = model.predict(X_test.values, verbose=0).ravel()
    nn_metrics_default = evaluate_and_plot("Neural Network", y_test, test_proba, 0.5, args.figures)
    nn_metrics_tuned = evaluate_and_plot(
        f"Neural Network (tuned threshold={best_threshold:.2f})", y_test, test_proba, best_threshold, args.figures)

    model.save(f"{args.out}/neural_network.keras")

    # --- Combine with Day 3 results into the FINAL comparison ---
    with open(f"{args.out}/baseline_model_comparison.json") as f:
        prior_results = json.load(f)
    # keep only the core rows (drop any previously-added NN rows on rerun)
    prior_results = [r for r in prior_results if "Neural Network" not in r["model"]]
    all_results = prior_results + [nn_metrics_default, nn_metrics_tuned]

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f"{args.out}/final_model_comparison.csv", index=False)
    with open(f"{args.out}/final_model_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}\nFINAL MODEL COMPARISON (all models)\n{'='*60}")
    print(results_df.to_string(index=False))

    # --- ROC curve: all three main models together ---
    dt = joblib.load(f"{args.out}/decision_tree.joblib")
    rf = joblib.load(f"{args.out}/random_forest.joblib")
    dt_proba = dt.predict_proba(X_test)[:, 1]
    rf_proba = rf.predict_proba(X_test)[:, 1]

    plt.figure(figsize=(6.5, 6.5))
    for name, proba in [("Decision Tree", dt_proba), ("Random Forest", rf_proba), ("Neural Network", test_proba)]:
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Final Model Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{args.figures}/roc_comparison_final.png", dpi=150)
    plt.close()

    # --- Final bar chart comparison (best variant of each model family) ---
    best_rows = {
        "Decision Tree": max([r for r in all_results if r["model"].startswith("Decision Tree")], key=lambda r: r["f1_score"]),
        "Random Forest": max([r for r in all_results if r["model"].startswith("Random Forest")], key=lambda r: r["f1_score"]),
        "Neural Network": max([r for r in all_results if r["model"].startswith("Neural Network")], key=lambda r: r["f1_score"]),
    }
    metrics_to_plot = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    x = np.arange(len(metrics_to_plot))
    width = 0.25
    plt.figure(figsize=(10, 5.5))
    for i, (name, row) in enumerate(best_rows.items()):
        plt.bar(x + i * width, [row.get(m, 0) or 0 for m in metrics_to_plot], width, label=name)
    plt.xticks(x + width, metrics_to_plot)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Final Model Comparison (best variant per model)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{args.figures}/final_model_comparison.png", dpi=150)
    plt.close()

    best_overall = max(best_rows.items(), key=lambda kv: kv[1]["f1_score"])
    print(f"\nBEST OVERALL MODEL (by F1): {best_overall[0]} -> {best_overall[1]}")

    joblib.dump(best_overall[0], f"{args.out}/best_model_name.joblib")
    print(f"\nSaved neural network to {args.out}/neural_network.keras")
    print(f"Saved final comparison to {args.out}/final_model_comparison.csv and .json")
    print(f"Saved figures to {args.figures}/")


if __name__ == "__main__":
    main()
