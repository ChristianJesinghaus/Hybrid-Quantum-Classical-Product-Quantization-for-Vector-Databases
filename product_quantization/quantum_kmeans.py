# -*- coding: utf-8 -*-
__author__ = 'Christian Jesinghaus'

# © 2025 Christian Jesinghaus
# SPDX-License-Identifier: LicenseRef-BA-Citation 

import json
import time
import logging
from typing import Optional, Dict, Any, List

import numpy as np
from qiskit import Aer

from .quantum_distance import QuantumDistanceCalculator, _ALLOWED_METRICS

#tqdm progress bar
try:
    from tqdm import trange, tqdm
except ImportError:
    def trange(x, **kw):  
        return range(x)
    def tqdm(x, *a, **k): 
        return x


class QuantumKMeans:
    """
    Quantum K‑Means with
       • re-weighted eigenvector candidates (IRLS / Rayleigh)
       • Safeguard: Backtracking along negative Riemann gradients (sphere)
       • k-means++ initialization
       • Reporting/history per iteration
    """
    #tolerance adapted from 2e-3 to 1e-2
    def __init__(self, n_clusters: int, max_iter: int = 100, shots: int = 1024,
                 tolerance: float = 1e-2, random_state: Optional[int] = None,
                 backend=None, distance_metric: str = "log_fidelity",
                 smooth_eps: float = 1e-3, **kwargs):
        dm = distance_metric.lower()
        if dm not in _ALLOWED_METRICS:
            raise ValueError(f"Unknown distance metric '{distance_metric}'. "
                             f"Allowed: {_ALLOWED_METRICS}")
        self.distance_metric = ("one_minus_fidelity"
                                if dm in ("one_minus_fidelity", "swap_test",
                                          "1-f", "omf")
                                else "log_fidelity")

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.shots = shots
        self.tolerance = tolerance
        self.random_state = random_state

        if backend is None:
            backend = Aer.get_backend('qasm_simulator')
        self.backend = backend
        self.smooth_eps = smooth_eps

        self._distance_calc = QuantumDistanceCalculator(
            shots=self.shots, backend=self.backend, smooth_eps=self.smooth_eps
        )

        # Outputs
        self.labels_: Optional[np.ndarray] = None
        self.cluster_centers_: Optional[np.ndarray] = None
        self.inertia_: Optional[float] = None

        # Reporting
        self.history_: List[Dict[str, Any]] = []
        self._last_dmat: Optional[np.ndarray] = None

    #   Distance & Assignment
    def _quantum_distance(self, x, c):
        return self._distance_calc.distance(x, c, metric=self.distance_metric)

    def _assign_clusters_quantum(self, X: np.ndarray) -> np.ndarray:
        # Cache the distance matrix for reporting/objective
        dmat = self._distance_calc.pairwise_distance_matrix(
            X, self.cluster_centers_, metric=self.distance_metric
        )
        self._last_dmat = dmat
        return np.argmin(dmat, axis=1)

    #   k‑means++ Initialization
    def _kmeans_pp_init(self, X: np.ndarray, rng: np.random.Generator):
        n = len(X)
        idx = [rng.integers(n)]
        while len(idx) < self.n_clusters:
            d2 = self._distance_calc.pairwise_distance_matrix(
                X, X[idx], metric=self.distance_metric
            ).min(axis=1)
            d2_sq = d2 ** 2
            tot = float(d2_sq.sum())
            if tot == 0.0:
                idx.extend(rng.choice(n, self.n_clusters - len(idx), replace=False))
                break
            probs = d2_sq / tot
            new_idx = rng.choice(n, p=probs)
            if new_idx not in idx:
                idx.append(int(new_idx))
        return np.array(idx, dtype=int)

    #   Helper functions (Loss, Grad, Objective, Separation measure)
    @staticmethod
    def _cluster_log_fid_loss(pts: np.ndarray, c: np.ndarray, eps: float) -> float:
        """f_j(c) = sum_i -log(|<psi_i|c>|^2 + eps)"""
        t = pts @ c
        s = np.abs(t) ** 2
        return float(-np.sum(np.log(s + eps)))

    def _compute_objective_from_assign(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Sum of distances to assigned centroids - uses cache if available."""
        if self._last_dmat is None or self._last_dmat.shape[0] != len(X):
            dmat = self._distance_calc.pairwise_distance_matrix(
                X, self.cluster_centers_, metric=self.distance_metric
            )
        else:
            dmat = self._last_dmat
        rows = np.arange(len(X))
        return float(np.sum(dmat[rows, labels]))

    @staticmethod
    def _min_offdiag_centroid_fid(centroids: np.ndarray) -> float:
        """ min_{i<j} |<phi_i|phi_j>|^2 (normalized centroids)."""
        if centroids is None or len(centroids) <= 1:
            return 1.0
        C = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
        G = np.abs(C @ C.T) ** 2
        m = np.min(G + np.eye(len(C)) * 2.0)  # Diagonale ausblenden (auf >1 setzen)
        return float(m)

    #    Safeguarded Centroid Update (with inner tqdm)
    def _centroid_update(self, X: np.ndarray, labels: np.ndarray):
        eps_num = 1e-12
        newC = np.zeros_like(self.cluster_centers_)

        accept_count = 0
        backtracks: List[int] = []
        grad_norms: List[float] = []
        cluster_sizes: List[int] = []
        cluster_losses: List[float] = []

        for k in trange(self.n_clusters, desc="Centroid‑update", leave=False):
            pts = X[labels == k]
            cluster_sizes.append(int(len(pts)))
            if len(pts) == 0:
                newC[k] = self.cluster_centers_[k]
                cluster_losses.append(0.0)
                backtracks.append(0)
                grad_norms.append(0.0)
                continue

            if self.distance_metric == "log_fidelity":
                c_old = self.cluster_centers_[k]

                # Re-weighted EV-Candidate
                overlaps = np.abs(pts @ c_old)
                F = overlaps ** 2
                w = 1.0 / (F + self.smooth_eps)
                w /= np.sum(w)
                Sigma = (pts.T * w) @ pts    # PSD, hermitian

                # Riemann gradient norm at old point (up to constant)
                Sc = Sigma @ c_old
                # Projection: (I - c c^T) Sc
                proj = Sc - (np.vdot(c_old, Sc).real) * c_old
                grad_norms.append(float(2.0 * np.linalg.norm(proj)))

                # Leading EV
                eigvals, eigvecs = np.linalg.eigh(Sigma)
                m = eigvecs[:, np.argmax(eigvals)]
                c_cand = m / (np.linalg.norm(m) + eps_num)

                f_old = self._cluster_log_fid_loss(pts, c_old, self.smooth_eps)
                f_cand = self._cluster_log_fid_loss(pts, c_cand, self.smooth_eps)

                if f_cand <= f_old:
                    newC[k] = c_cand
                    accept_count += 1
                    backtracks.append(0)
                else:
                    #  Backtracking at negative Riemann gradient
                    d = proj
                    d_norm = np.linalg.norm(d)
                    if d_norm < 1e-12:
                        newC[k] = c_old
                        backtracks.append(0)
                    else:
                        d = d / d_norm
                        step = 1.0
                        bt = 0
                        accepted = False
                        for _ in range(10):  # 10 Halvings set
                            bt += 1
                            c_try = c_old + step * d
                            c_try /= (np.linalg.norm(c_try) + eps_num)
                            f_try = self._cluster_log_fid_loss(pts, c_try, self.smooth_eps)
                            if f_try < f_old:
                                newC[k] = c_try
                                accepted = True
                                break
                            step *= 0.5
                        if accepted:
                            backtracks.append(bt)
                        else:
                            newC[k] = c_old
                            backtracks.append(bt)

                # Loss of new centroid (for Reporting)
                cluster_losses.append(self._cluster_log_fid_loss(pts, newC[k], self.smooth_eps))

            else:
                # one_minus_fidelity: classical Mean + Normalizing
                m = np.mean(pts, axis=0)
                c_new = m / (np.linalg.norm(m) + eps_num)
                newC[k] = c_new
                cluster_losses.append(0.0)
                backtracks.append(0)
                grad_norms.append(0.0)

        stats = {
            "accept_count": accept_count,
            "backtracks": backtracks,
            "grad_norms": grad_norms,
            "cluster_sizes": cluster_sizes,
            "cluster_losses": cluster_losses,
        }
        return newC, stats

    
    #   Fit‑Routine
    def fit(self, X: np.ndarray):
        rng = np.random.default_rng(self.random_state)
        self.cluster_centers_ = X[self._kmeans_pp_init(X, rng)]

        logger = logging.getLogger(__name__)
        self.history_.clear()

        for it in trange(self.max_iter, desc="QuantumKMeans"):
            t0 = time.time()

            # Step 1: Assignment 
            labels = self._assign_clusters_quantum(X)
            objective_before = self._compute_objective_from_assign(X, labels)

            # Step 2: Centroid-Update with Safeguard 
            new_centers, stats = self._centroid_update(X, labels)
            shift = float(np.linalg.norm(new_centers - self.cluster_centers_))
            self.cluster_centers_ = new_centers

            # Objective after Update
            # (Fresh distances to new centroids)
            dmat_after = self._distance_calc.pairwise_distance_matrix(
                X, self.cluster_centers_, metric=self.distance_metric
            )
            self._last_dmat = dmat_after  # Cache for possible next round
            objective_after = float(np.sum(dmat_after[np.arange(len(X)), labels]))

            # Iterations-Logging
            accept_ratio = (
                stats["accept_count"] / self.n_clusters if self.distance_metric == "log_fidelity" else 1.0
            )
            bt_mean = float(np.mean(stats["backtracks"])) if stats["backtracks"] else 0.0
            grad_max = float(np.max(stats["grad_norms"])) if stats["grad_norms"] else 0.0
            min_fid = self._min_offdiag_centroid_fid(self.cluster_centers_)
            iter_time = float(time.time() - t0)

            self.history_.append({
                "iter": it + 1,
                "objective_before": objective_before,
                "objective_after": objective_after,
                "shift": shift,
                "accept_ratio": accept_ratio,
                "backtracks_mean": bt_mean,
                "grad_norm_max": grad_max,
                "cluster_size": stats["cluster_sizes"],
                "cluster_loss": stats["cluster_losses"],
                "min_centroid_fid_offdiag": min_fid,
                "iter_time_sec": iter_time,
            })

            if shift < self.tolerance:
                logger.info("QuantumKMeans converged in %d iterations (shift=%.3e)", it + 1, shift)
                break

        # Final Assignment + Inertia
        self.labels_ = self._assign_clusters_quantum(X)
        self.inertia_ = sum(
            self._quantum_distance(X[i], self.cluster_centers_[self.labels_[i]])
            for i in range(len(X))
        )
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.labels_

    #   Export of History
    def export_history(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.history_, f, indent=2, ensure_ascii=False)
