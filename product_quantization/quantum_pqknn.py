# -*- coding: utf-8 -*-
__author__ = "Christian Jesinghaus"
 
# © 2025 Christian Jesinghaus
# SPDX-License-Identifier: LicenseRef-BA-Citation


import os
import json
import numpy as np
from typing import Optional, Dict, Any

import logging
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, *a, **k):  # noqa
        return x

from .quantum_kmeans import QuantumKMeans
from .quantum_distance import QuantumDistanceCalculator, _ALLOWED_METRICS
from .util import log_nb_clusters_to_np_int_type

logger = logging.getLogger(__name__)


class QuantumProductQuantizationKNN:
    """Hybrid / Quantum Product‑Quantization kNN (immer Quantum‑Pfad)."""

    def __init__(
        self,
        n: int,
        c: int,
        *,
        max_iter_qk: int = 15,
        quantum_shots: int = 1024,
        random_state: Optional[int] = None,
        distance_metric: str = "log_fidelity",
        smooth_eps: float = 1e-3,
        **kwargs,
    ):
        #validate metric
        dm = distance_metric.lower()
        if dm not in _ALLOWED_METRICS:
            raise ValueError(
                f"Unknown distance metric '{distance_metric}'. Allowed: {_ALLOWED_METRICS}"
            )
        self.distance_metric = (
            "one_minus_fidelity"
            if dm in ("one_minus_fidelity", "swap_test", "1-f", "omf")
            else "log_fidelity"
        )

        #Hyper‑Parameter
        self.n = n
        self.c = c
        self.max_iter_qk = max_iter_qk
        self.quantum_shots = quantum_shots
        self.random_state = random_state
        self.smooth_eps = smooth_eps

        #  Quantum Helper 
        self._distance_calc = QuantumDistanceCalculator(
            shots=self.quantum_shots, smooth_eps=self.smooth_eps
        )

        bits = int(np.ceil(np.log2(max(1, self.c))))
        self.int_type = log_nb_clusters_to_np_int_type(bits)

        # -Placeholders
        self.partition_size: Optional[int] = None
        self.d: Optional[int] = None
        self.compressed_data: Optional[np.ndarray] = None
        self.subvector_centroids: Dict[int, np.ndarray] = {}
        self.train_labels: Optional[np.ndarray] = None

        # Trainings‑Histories of QuantumKMeans‑Parts per Partition
        self.subvector_histories: Dict[int, Any] = {}

    # helpers
    def _get_data_partition(self, data: np.ndarray, idx: int) -> np.ndarray:
        s = idx * self.partition_size
        e = (idx + 1) * self.partition_size
        return data[:, s:e]

    def _compress_partition(self, p_idx: int, Xp: np.ndarray):
        
        qkm = QuantumKMeans(
            n_clusters=self.c,
            max_iter=self.max_iter_qk,
            shots=self.quantum_shots,
            random_state=self.random_state,
            distance_metric=self.distance_metric,
            smooth_eps=self.smooth_eps,
        )
        labels = qkm.fit_predict(Xp).astype(self.int_type)
        cents = qkm.cluster_centers_
        self.subvector_histories[p_idx] = getattr(qkm, "history_", None)
        return p_idx, labels, cents

    # API
    def compress(self, X: np.ndarray, y: np.ndarray):
        self.d = X.shape[1]
        self.partition_size = self.d // self.n
        if self.d % self.n:
            logger.warning("Dimension %d not divisible by n=%d.", self.d, self.n)

        self.compressed_data = np.zeros((len(X), self.n), dtype=self.int_type)
        self.train_labels = y

        for p in tqdm(range(self.n), desc="Compress partitions", unit="part"):
            # last partition takes remaining dimensions
            part = X[:, p * self.partition_size :] if p == self.n - 1 else self._get_data_partition(X, p)
            _, comp_part, cents = self._compress_partition(p, part)
            self.compressed_data[:, p] = comp_part
            self.subvector_centroids[p] = cents

        logger.info("Compression finished (ratio %.3f).", self.get_compression_ratio())

    def _partition_dists(self, sample: np.ndarray) -> np.ndarray:
        D = np.zeros((self.c, self.n))
        for p in range(self.n):
            s_part = sample[p * self.partition_size :] if p == self.n - 1 else sample[
                p * self.partition_size : (p + 1) * self.partition_size
            ]
            cents = self.subvector_centroids[p]
            D[:, p] = self._distance_calc.pairwise_distance_matrix(
                s_part.reshape(1, -1), cents, metric=self.distance_metric
            )[0]
        return D

    def _predict_one(self, sample: np.ndarray, k: int) -> int:
        Dp = self._partition_dists(sample)
        d_sum = np.zeros(len(self.compressed_data))
        for p in range(self.n):
            d_sum += Dp[:, p][self.compressed_data[:, p]]

        nn_idx = np.argsort(d_sum)[:k]
        vals, cnts = np.unique(self.train_labels[nn_idx], return_counts=True)
        return vals[np.argmax(cnts)]

    def predict(self, X: np.ndarray, k: int = 1) -> np.ndarray:
        if self.compressed_data is None:
            raise RuntimeError("Model not compressed") 
        return np.array([self._predict_one(x, k) for x in X], dtype=self.train_labels.dtype)


    # utility
    def get_compression_ratio(self) -> float:
        if self.compressed_data is None:
            return 0.0
        return (self.d * len(self.compressed_data)) / np.prod(self.compressed_data.shape)

    def get_quantum_info(self) -> dict:
        info = {
            "metric": self.distance_metric,
            "shots": self.quantum_shots,
            "smooth_eps": self.smooth_eps,
        }
        if hasattr(self, "_distance_calc") and hasattr(self._distance_calc, "get_stats"):
            try:
                info["distance_stats"] = self._distance_calc.get_stats()
            except Exception:
                pass
        return info

    def export_histories(self, dir_path: str):
        os.makedirs(dir_path, exist_ok=True)
        for p, hist in self.subvector_histories.items():
            if hist is None:
                continue
            out = os.path.join(dir_path, f"partition_{p:02d}_qkm_history.json")
            with open(out, "w", encoding="utf-8") as f:
                json.dump(hist, f, indent=2, ensure_ascii=False)
