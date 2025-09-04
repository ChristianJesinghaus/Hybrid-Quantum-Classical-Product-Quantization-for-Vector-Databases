# -*- coding: utf-8 -*-
__author__ = "Christian Jesinghaus"
# © 2025 Christian Jesinghaus 
# SPDX-License-Identifier: LicenseRef-BA-Citation

import os
import time
import traceback
from datetime import datetime
from typing import Dict, Any

import numpy as np
from sklearn.metrics import classification_report

from product_quantization.experiment_utils import generate_experiment_name
from product_quantization.quantum_pqknn import QuantumProductQuantizationKNN
from product_quantization.PQKNN import ProductQuantizationKNN  # CHANGED: klassisches Modell
from product_quantization.txt_config_loader import ConfigLoader
from product_quantization.model_persistence import ModelPersistence
from product_quantization.experiment_utils import (
    print_evaluation_summary,
    generate_experiment_name,
)
from product_quantization.normalize import normalize_data
from product_quantization.visualization import plot_confusion_matrix  

#   Data loading and normalisation
def load_and_prepare(config: Dict[str, Any]):
    path = config.get("data_file", "example_data.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with np.load(path) as d:
        train_x, train_y = d["train_data"], d["train_labels"]
        test_x, test_y = d["test_data"], d["test_labels"]

    train_x, train_y = train_x[: config["train_size"]], train_y[: config["train_size"]]
    test_x, test_y = test_x[: config["test_size"]], test_y[: config["test_size"]]

    if config.get("normalize_data", False):
        train_x, test_x = normalize_data(train_x), normalize_data(test_x)
    return train_x, train_y, test_x, test_y


def sanitize_metric(metric_raw: str | None) -> str:
    """Sanitizes everything inside "" and after '#'."""
    if not metric_raw:
        return "log_fidelity"
    metric = metric_raw.split("#", 1)[0].strip()
    return metric.strip("'\"")

def main():
    print("Hybrid / Classical Product Quantization KNN\n")
    cfg_loader = ConfigLoader("config.txt")
    cfg = cfg_loader.load_config()

    cfg["distance_metric"] = sanitize_metric(cfg.get("distance_metric"))

    # print config
    for k, v in cfg.items():
        print(f"{k:25}: {v}")
    print()

    try:
        train_x, train_y, test_x, test_y = load_and_prepare(cfg)

        algo = (cfg.get("algorithm") or "quantum").strip().lower()
        if algo == "classical":
            k_clusters_cfg = cfg.get("k_clusters")
            # if k_clusters is set, use it, otherwise fall back to c
            model = ProductQuantizationKNN(
                n=cfg["n"],
                c=0,
                k_clusters=k_clusters_cfg if k_clusters_cfg is not None else cfg["c"],
                random_state=cfg.get("random_state", 42)
            )

        else:
            model = QuantumProductQuantizationKNN(
                n=cfg["n"],
                c=cfg["c"],
                max_iter_qk=cfg.get("max_iter_qk", 15),
                quantum_shots=cfg.get("quantum_shots", 1024),
                random_state=cfg.get("random_state", 42),
                distance_metric=cfg.get("distance_metric", "log_fidelity"),
                smooth_eps=cfg.get("log_fidelity_precision", 1e-3),
            )

        # Compression
        t0 = time.time()
        model.compress(train_x, train_y)
        comp_t = time.time() - t0
        print(f"Compression: {comp_t:.2f}s (ratio {model.get_compression_ratio():.2f}×)")

        # Export histories (both models have export_histories hook)
        exp_name = generate_experiment_name(cfg)  
        metric = cfg.get("distance_metric", "log_fidelity")
        eps = cfg.get("log_fidelity_precision", 1e-3)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = f"experiments/histories/{exp_name}_{metric}_eps{eps:g}_{ts}"
        if hasattr(model, "export_histories"):
            model.export_histories(outdir)
            print(f"[INFO] Histories saved to {outdir}")

        # --- Prediction
        t0 = time.time()
        preds = model.predict(test_x, k=cfg["k"])
        pred_t = time.time() - t0
        acc = print_evaluation_summary(preds, test_y, verbose=cfg.get("verbose", True))
        print(f"Prediction:  {pred_t:.2f}s")

        # Optional: Confusion matrix / Classification report saving
        report_dir = cfg.get("experiment_report_dir", "experiments/reports")
        os.makedirs(report_dir, exist_ok=True)
        tag = f"{algo}_n{cfg['n']}_c{cfg['c']}_k{cfg['k']}_{ts}"

        if cfg.get("save_confusion_matrix", True):
            try:
                fig = plot_confusion_matrix(test_y, preds, classes=None,
                                            title=f"Confusion Matrix ({tag})")
                cm_path = os.path.join(report_dir, f"confusion_matrix_{tag}.png")
                fig.savefig(cm_path, dpi=140, bbox_inches="tight")
                print(f"[INFO] Confusion matrix saved to {cm_path}")
            except Exception as e:
                print(f"[WARN] could not save confusion matrix: {e}")

        if cfg.get("save_classification_report", True):
            try:
                cr = classification_report(test_y, preds, digits=4)
                cr_path = os.path.join(report_dir, f"classification_report_{tag}.txt")
                with open(cr_path, "w", encoding="utf-8") as f:
                    f.write(cr)
                print(f"[INFO] Classification report saved to {cr_path}")
            except Exception as e:
                print(f"[WARN] could not save classification report: {e}")

        #  Safe
        results = {
            "algorithm": algo,
            "accuracy": float(acc),
            "train_compress_time_s": float(comp_t),
            "predict_time_s": float(pred_t),
            "compression_ratio": float(model.get_compression_ratio()),
            "n": int(cfg["n"]),
            "c": int(cfg["c"]),
            "k_neighbors": int(cfg["k"]),
            "distance_metric": cfg.get("distance_metric", "log_fidelity"),
            "quantum_shots": int(cfg.get("quantum_shots", 0)),
        }

        if cfg.get("save_model", True):
            ModelPersistence(cfg.get("model_output_dir", "experiments/models")).save_model(
                model, cfg, results
            )

    except Exception as e:
        print("[ERROR]", e)
        traceback.print_exc()


if __name__ == "__main__":
    main()
