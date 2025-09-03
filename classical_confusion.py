# © 2025 Christian Jesinghaus
# SPDX-License-Identifier: LicenseRef-BA-Citation



"""
Script to generate a normalized confusion matrix for a saved classical PQKNN model.

Usage example:
    python confusion_classical.py --model_dir experiments/models/pqknn_n8_c10_classical_20250825_121224

Only classical runs are processed.
"""

import argparse
import json
import os
from typing import Any, Dict

import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from product_quantization.model_persistence import ModelPersistence
from product_quantization.normalize import normalize_data
from product_quantization.PQKNN import ProductQuantizationKNN  # für Fallback
from product_quantization.visualization import plot_confusion_matrix

def load_dataset(cfg: Dict[str, Any]):
    """Load and prepare the dataset according to the given config."""
    data_path = cfg.get("data_file", "example_data.npz")
    if not os.path.isfile(data_path):
        alt_path = os.path.join(os.path.dirname(cfg.get("config_path", "")), "..", "..", data_path)
        if os.path.isfile(alt_path):
            data_path = alt_path
        else:
            raise FileNotFoundError(f"Dataset file {data_path} not found.")
    with np.load(data_path) as d:
        train_x, train_y = d["train_data"], d["train_labels"]
        test_x, test_y = d["test_data"], d["test_labels"]
    train_size = cfg.get("train_size", len(train_x))
    test_size = cfg.get("test_size", len(test_x))
    train_x, train_y = train_x[:train_size], train_y[:train_size]
    test_x, test_y = test_x[:test_size], test_y[:test_size]
    if cfg.get("normalize_data", False):
        train_x, test_x = normalize_data(train_x), normalize_data(test_x)
    return train_x, train_y, test_x, test_y

def generate_confusion(model_dir: str, k: int = None, save: bool = True):
    cfg_path = os.path.join(model_dir, "config.json")
    model_path = os.path.join(model_dir, "model.pkl")
    if not os.path.isfile(cfg_path) or not os.path.isfile(model_path):
        print(f"[ERROR] Missing config.json or model.pkl in {model_dir}")
        return
    with open(cfg_path) as f:
        cfg = json.load(f)
    algo = cfg.get("algorithm")
    if not algo:
        if cfg.get("use_quantum_training") or cfg.get("use_quantum_prediction"):
            algo = "quantum"
        else:
            algo = "classical"
    if algo.lower() != "classical":
        print(f"[WARN] The selected run appears to be quantum (algorithm={algo}). This script handles only classical models.")
        return

    # Load model via ModelPersistence
    persistence = ModelPersistence(os.path.dirname(model_dir))
    model, cfg_loaded, _ = persistence.load_model(model_dir)
    if model is None:
        print(f"[ERROR] Could not load model from {model_dir}")
        return
    cfg_loaded["config_path"] = cfg_path

    # Load full training and test sets
    train_x, train_y, X_test, y_test = load_dataset(cfg_loaded)
    k_cfg = cfg_loaded.get("k", 3)
    k_val = k if k is not None else k_cfg

    try:
        # Try to predict using the stored model
        preds = model.predict(X_test, k=k_val)
    except Exception as exc:
        # If it fails (e.g. mismatched centroids), train a fresh model
        print(f"[WARN] Prediction failed: {exc}")
        print("[INFO] Training a new ProductQuantizationKNN model for fallback…")
        n_partitions = cfg_loaded.get("n")
        k_clusters = cfg_loaded.get("k_clusters")
        if not k_clusters:
            c_exp = cfg_loaded.get("c")
            k_clusters = 2 ** int(c_exp) if c_exp is not None else 2 ** 10
        random_state = cfg_loaded.get("random_state", 42)
        fresh_model = ProductQuantizationKNN(n=n_partitions, k_clusters=k_clusters, random_state=random_state)
        fresh_model.compress(train_x, train_y)
        preds = fresh_model.predict(X_test, k=k_val)

    # Normalise confusion matrix per class
    cm = confusion_matrix(y_test, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    classes = [str(i) for i in np.unique(y_test)]

    # Plot normalised confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"Normalized confusion matrix (Classical, M={len(X_test)})")
    plt.tight_layout()

    if save:
        out_path = os.path.join(model_dir, f"confusion_matrix_M{len(X_test)}.png")
        plt.savefig(out_path)
        print(f"[INFO] Confusion matrix saved to {out_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Generate a normalized confusion matrix for a classical PQKNN model")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to the model directory containing config.json and model.pkl")
    parser.add_argument("--k", type=int, default=None,
                        help="Number of neighbours to use; default is value from config")
    parser.add_argument("--no_save", action="store_true",
                        help="Do not save the plot, display it interactively instead")
    args = parser.parse_args()
    generate_confusion(args.model_dir, k=args.k, save=not args.no_save)

if __name__ == "__main__":
    main()
