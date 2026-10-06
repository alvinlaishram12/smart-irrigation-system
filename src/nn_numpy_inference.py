"""
nn_numpy_inference.py
----------------------
Day 8 finding: importing TensorFlow alone consumes ~700MB RAM (measured),
dangerously close to Streamlit Community Cloud's 1GB free-tier ceiling once
pandas/scikit-learn/Streamlit's own overhead are added. The Day 7 "best
model" (Neural Network, F1=0.724) would likely crash the deployed app on
the free tier purely from TensorFlow's import cost - before it even serves
a single prediction.

The network itself is tiny (15 inputs -> Dense(32,relu) -> Dense(16,relu) ->
Dense(1,sigmoid); Dropout is a no-op at inference time), so rather than
downgrade to a worse model just to save memory, this extracts the trained
weights once and reimplements the forward pass in plain NumPy. This keeps
the exact model Day 7 selected - same weights, same predictions, verified
numerically identical - while removing a ~700MB dependency from the
deployed app's import path entirely. TensorFlow remains a (dev-only)
dependency for *training* the network; it is never imported by the
deployed dashboard or predict_live.py.

Usage:
    python src/nn_numpy_inference.py --export    # run once after training
    python src/nn_numpy_inference.py --verify    # confirm numpy == keras
"""

import argparse

import numpy as np


def export_weights(keras_model_path="models/neural_network.keras",
                    out_path="models/neural_network_weights.npz"):
    """One-time export. Only this function ever imports TensorFlow."""
    import tensorflow as tf  # local import: keeps TF out of the deploy path

    model = tf.keras.models.load_model(keras_model_path)
    dense_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]
    if len(dense_layers) != 3:
        raise ValueError(f"Expected 3 Dense layers, found {len(dense_layers)}. "
                          "Update export_weights() if the architecture changed.")

    weights = {}
    for i, layer in enumerate(dense_layers, start=1):
        W, b = layer.get_weights()
        weights[f"W{i}"] = W
        weights[f"b{i}"] = b

    np.savez(out_path, **weights)
    print(f"Exported {sum(w.size for w in weights.values())} parameters to {out_path}")
    return weights


def load_numpy_model(weights_path="models/neural_network_weights.npz"):
    data = np.load(weights_path)
    return {k: data[k] for k in data.files}


def _relu(x):
    return np.maximum(0, x)


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def predict_proba_numpy(X, weights):
    """X: (n_samples, 15) already-scaled feature matrix, same column order
    used at training time. Returns P(irrigate=1) per row, matching
    keras_model.predict(X).ravel() exactly (within float precision)."""
    X = np.asarray(X, dtype=np.float64)
    h1 = _relu(X @ weights["W1"] + weights["b1"])
    h2 = _relu(h1 @ weights["W2"] + weights["b2"])
    out = _sigmoid(h2 @ weights["W3"] + weights["b3"])
    return out.ravel()


def verify_against_keras(data_dir="data/processed", models_dir="models"):
    """Loads the real Keras model fresh and compares its output to the
    NumPy reimplementation on the actual held-out test set."""
    import pandas as pd
    import tensorflow as tf

    X_test = pd.read_csv(f"{data_dir}/X_test.csv").values
    keras_model = tf.keras.models.load_model(f"{models_dir}/neural_network.keras")
    keras_probs = keras_model.predict(X_test, verbose=0).ravel()

    weights = load_numpy_model(f"{models_dir}/neural_network_weights.npz")
    numpy_probs = predict_proba_numpy(X_test, weights)

    max_abs_diff = np.max(np.abs(keras_probs - numpy_probs))
    # Also confirm binary decisions never differ at the tuned threshold
    import joblib
    threshold = joblib.load(f"{models_dir}/tuned_thresholds.joblib").get("neural_network", 0.5)
    keras_decisions = keras_probs >= threshold
    numpy_decisions = numpy_probs >= threshold
    decisions_match = np.array_equal(keras_decisions, numpy_decisions)

    print(f"Test set size: {len(X_test)}")
    print(f"Max absolute probability difference: {max_abs_diff:.2e}")
    print(f"All binary decisions match at threshold {threshold:.3f}: {decisions_match}")
    if max_abs_diff > 1e-5 or not decisions_match:
        raise AssertionError("NumPy reimplementation diverges from Keras - do not deploy this.")
    print("VERIFIED: NumPy inference is numerically identical to the trained Keras model.")


def main():
    parser = argparse.ArgumentParser(description="Export/verify NumPy-only NN inference.")
    parser.add_argument("--export", action="store_true", help="Export weights from the .keras model")
    parser.add_argument("--verify", action="store_true", help="Verify NumPy matches Keras on the test set")
    args = parser.parse_args()

    if args.export:
        export_weights()
    if args.verify:
        verify_against_keras()
    if not args.export and not args.verify:
        parser.print_help()


if __name__ == "__main__":
    main()
