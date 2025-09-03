# -*- coding: utf-8 -*-
__author__ = "Christian Jesinghaus"

# © 2025 Christian Jesinghaus
# SPDX-License-Identifier: LicenseRef-BA-Citation

import os
import glob
import json
from typing import Any, Dict
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

from product_quantization.model_persistence import ModelPersistence
from product_quantization.normalize import normalize_data


def load_dataset(cfg: Dict[str, Any]):
    data_path = cfg.get("data_file", "example_data.npz")
    if not os.path.exists(data_path):
        raise FileNotFoundError(data_path)

    with np.load(data_path) as d:
        train_x, train_y = d["train_data"], d["train_labels"]
        test_x, test_y = d["test_data"], d["test_labels"]

    train_size = cfg.get("train_size", len(train_x))
    test_size = cfg.get("test_size", len(test_x))
    train_x, train_y = train_x[:train_size], train_y[:train_size]
    test_x, test_y = test_x[:test_size], test_y[:test_size]

    if cfg.get("normalize_data", False):
        train_x, test_x = normalize_data(train_x), normalize_data(test_x)

    return test_x, test_y


def main():
    models_dir = "experiments/models"
    model_files = glob.glob(f"{models_dir}/**/*model.pkl", recursive=True)
    if not model_files:
        print(f"[ERROR] No models found under {models_dir}.")
        return

    print("Available models:")
    for i, m in enumerate(model_files):
        print(f"[{i}] {os.path.relpath(m, models_dir)}")

    sel = int(input("Select model number: "))
    model_path = model_files[sel]

    persistence = ModelPersistence(models_dir)
    model, cfg, results = persistence.load_model(os.path.dirname(model_path))
    if model is None:
        return

    X_test, y_test = load_dataset(cfg)
    print(f"Test set size according to config: {len(X_test)}")

    k_def = cfg.get("k", 3)
    k = int(input(f"k neighbors [{k_def}]: ") or k_def)

    algo = str(cfg.get("algorithm", "quantum")).lower()
    metric = cfg.get("distance_metric", "log_fidelity")

    
    if algo == 'quantum':
        user_metric = input(f"Distance metric for Quantum [{metric}]: ").strip().lower()
        if user_metric:
            metric = user_metric
        
        if hasattr(model, "distance_metric"):
            model.distance_metric = metric

    print(f"[INFO] predict | algo={algo}, k={k}"
          + (f", metric={metric}" if algo == 'quantum' else "")
          + f", Test set size={len(X_test)}")

    preds = model.predict(X_test, k=k)

    acc = accuracy_score(y_test, preds)
    print("\n" + "=" * 50)
    print(f"Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, preds, digits=4))
    cm = confusion_matrix(y_test, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues')
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"Confusion matrix for {os.path.basename(model_path)}")
    plt.tight_layout()
    plt.show()
    if isinstance(results, dict) and "accuracy" in results:
        ref = results["accuracy"]
        delta = acc - ref
        print(f"\n Reference accuracy according to summary: {ref*100:.2f}% | Δ={delta*100:+.2f}%")

if __name__ == "__main__":
    main()
