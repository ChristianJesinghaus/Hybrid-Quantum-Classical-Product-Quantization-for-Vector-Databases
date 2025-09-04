# -*- coding: utf-8 -*-
__author__ = "Christian Jesinghaus"
 
# © 2025 Christian Jesinghaus
# SPDX-License-Identifier: LicenseRef-BA-Citation

import numpy as np
from typing import List, Optional
import logging
from qiskit import (
    QuantumCircuit,
    QuantumRegister,
    ClassicalRegister,
    transpile,
    execute,
    Aer,
)
try:
    from tqdm import trange
except ImportError:  
    def trange(*args, **kwargs):
        return range(*args, **kwargs)

logger = logging.getLogger(__name__)

_ALLOWED_METRICS = {
    "log_fidelity",
    "one_minus_fidelity",
    "swap_test",
    "1-f",
    "lf",
    "logf",
    "omf",
}


class QuantumDistanceCalculator:
    """
    Unified quantum distance calculator mit Smooth‑Clamping:
        d_log(F) = -log(F + eps)
    """
    def __init__(self, shots: int = 1024, backend=None, smooth_eps: float = 1e-3):
        self.shots = shots
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.smooth_eps = smooth_eps

    def distance(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray,
        *,
        mode: str | None = None,
        metric: str | None = None,
    ) -> float:
        """
        Calculates distance between two vectors

        accepts "mode" and "metric" for legacy reasons
        """
        #Legacy
        if mode is None:
            mode = metric or "log_fidelity"

        mode = self._normalize_mode(mode)
        F = self._fidelity(vec1, vec2)
        return self._smooth_log_distance(F) if mode == "log_fidelity" else 1.0 - F

    def fidelity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        return self._fidelity(vec1, vec2)

    #helpers 
    def _smooth_log_distance(self, F: float) -> float:
        return -np.log(min(1.0, F) + self.smooth_eps)

    def _normalize_mode(self, mode: str) -> str:
        mode = mode.lower()
        if mode not in _ALLOWED_METRICS:
            raise ValueError(f"Unknown distance mode '{mode}'. Allowed: {_ALLOWED_METRICS}")
        return (
            "one_minus_fidelity"
            if mode in ("one_minus_fidelity", "swap_test", "1-f", "omf")
            else "log_fidelity"
        )

    #Fidelity via Swap‑Test or classical-Fallback
    
    def _fidelity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        v1 = vec1 / (np.linalg.norm(vec1) + 1e-12)
        v2 = vec2 / (np.linalg.norm(vec2) + 1e-12)
        try:
            qc = self._create_swap_test_circuit(v1, v2)
            if qc is None:
                raise RuntimeError("Circuit creation failed")
            tqc = transpile(qc, self.backend)
            job = execute(tqc, self.backend, shots=self.shots)
            counts = job.result().get_counts()
            prob_0 = counts.get("0", 0) / self.shots
            overlap_sq = max(0.0, min(1.0, 2 * prob_0 - 1))
            return overlap_sq
        except Exception as e:
            logger.warning("Falling back to classical fidelity: %s", e)
            return float(np.abs(np.dot(v1, v2)) ** 2)

    def _create_swap_test_circuit(
        self, vec1: np.ndarray, vec2: np.ndarray
    ) -> Optional[QuantumCircuit]:
        try:
            from .util import amplitude_encoding

            d = len(vec1)
            n_qubits = int(np.ceil(np.log2(d))) if d > 1 else 1
            qreg1 = QuantumRegister(n_qubits, "v1")
            qreg2 = QuantumRegister(n_qubits, "v2")
            anc = QuantumRegister(1, "anc")
            creg = ClassicalRegister(1, "c")
            qc = QuantumCircuit(qreg1, qreg2, anc, creg)

            qc.compose(amplitude_encoding(vec1), qreg1, inplace=True)
            qc.compose(amplitude_encoding(vec2), qreg2, inplace=True)

            qc.h(anc[0])
            for i in range(n_qubits):
                qc.cswap(anc[0], qreg1[i], qreg2[i])
            qc.h(anc[0])
            qc.measure(anc[0], creg[0])
            return qc
        except Exception as e:
            logger.warning("Failed to create swap‑test circuit: %s", e)
            return None

    #Distancelist (Test‑Vektor vs. List)
    def quantum_distance_matrix(
        self,
        vectors: List[np.ndarray],
        test_vector: np.ndarray,
        mode: str = "log_fidelity",
    ) -> np.ndarray:
        mode = self._normalize_mode(mode)
        F_list = [self._fidelity(test_vector, v) for v in vectors]
        if mode == "log_fidelity":
            return np.array([self._smooth_log_distance(F) for F in F_list])
        return np.array([1.0 - F for F in F_list])

    #Alias for K‑Means / PQ‑kNN
    def pairwise_distance_matrix(
        self,
        X: List[np.ndarray],
        Y: Optional[List[np.ndarray]] = None,
        metric: str = "log_fidelity",
    ) -> np.ndarray:
        return quantum_pairwise_distances(
            np.array(X),
            np.array(Y) if Y is not None else None,
            metric,
            shots=self.shots,
            smooth_eps=self.smooth_eps,
        )


#External Helpers: full Pairwise Distance Matrix
def quantum_pairwise_distances(
    X: np.ndarray,
    Y: Optional[np.ndarray] = None,
    metric: str = "swap_test",
    *,
    shots: int = 1024,
    smooth_eps: float = 1e-3,
) -> np.ndarray:
    calc = QuantumDistanceCalculator(shots=shots, smooth_eps=smooth_eps)
    metric = calc._normalize_mode(metric)
    if Y is None:
        Y = X
    D = np.zeros((len(X), len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            F = calc.fidelity(X[i], Y[j])
            D[i, j] = (
                calc._smooth_log_distance(F) if metric == "log_fidelity" else 1.0 - F
            )
    return D
