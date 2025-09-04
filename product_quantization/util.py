# -*- coding: utf-8 -*-
__author__ = 'Jeroen Van Der Donckt'

 
# © 2021 Jeroen Van Der Donckt — License: MIT (see THIRD_PARTY_LICENSES/classical_PQKNN-MIT.txt)
# Modifications © 2025 Christian Jesinghaus
# SPDX-License-Identifier: MIT


import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, transpile, execute
from qiskit.circuit.library import QFT, GroverOperator
from qiskit.algorithms import AmplificationProblem
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector
    

try:
    from qiskit import QuantumCircuit
except ImportError:
    QuantumCircuit = None


def log_nb_clusters_to_np_int_type(log_nb_clusters: int) -> type:
    """
    Returns the appropriate numpy data type for cluster indices,
    based on the bit count log_nb_clusters.
    """
    if log_nb_clusters <= 8:
        return np.uint8
    elif log_nb_clusters <= 16:
        return np.uint16
    elif log_nb_clusters <= 32:
        return np.uint32
    else:
        return np.uint64


def squared_euclidean_dist(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """
    Computes the squared Euclidean distance
    """
    return np.sum(np.square(vec2 - vec1), axis=-1)


def amplitude_encoding(vector: np.ndarray) -> QuantumCircuit:
    """
    Implements quantum amplitude encoding.
    """
    
    # Normalization of the quantum state
    normalized = vector / (np.linalg.norm(vector) + 1e-12)

    # Pad to next power of 2 for quantum register
    n_qubits = int(np.ceil(np.log2(len(normalized))))
    padded_size = 2**n_qubits

    # Create complex-valued state vector
    padded = np.zeros(padded_size, dtype=complex)
    padded[:len(normalized)] = normalized.astype(complex)

    # Renormalization after padding
    padded = padded / np.linalg.norm(padded)

    # Create quantum circuit
    qc = QuantumCircuit(n_qubits)

    #  Real Quantum State Preparation
    state_prep = StatePreparation(padded)
    qc.append(state_prep, range(n_qubits))
    
    return qc





# The below given Functions are currently not used and just implemented as 
# an example or starting point for future work on them. This is also mentioned
#in the thesis.


def quantum_minimum_finding_oracle(distances: np.ndarray, threshold: float) -> QuantumCircuit:
    """
    Implementation of a possible quantum oracle for minimum finding.
    """
    n_items = len(distances)
    n_qubits = int(np.ceil(np.log2(n_items)))
    
    # Quantum Register
    qreg = QuantumRegister(n_qubits, 'search')
    qreg_ancilla = QuantumRegister(1, 'oracle')
    
    qc = QuantumCircuit(qreg, qreg_ancilla)

    # Mark all indices with distance below threshold
    for i, distance in enumerate(distances):
        if distance < threshold:
            # Binary representation of the index
            binary_repr = format(i, f'0{n_qubits}b')
            
            # X-Gates for 0-bits
            for j, bit in enumerate(binary_repr):
                if bit == '0':
                    qc.x(qreg[j])
            
            # Multi-controlled NOT on Ancilla
            if n_qubits == 1:
                qc.cx(qreg[0], qreg_ancilla[0])
            else:
                qc.mct(list(qreg), qreg_ancilla[0])

            # Z-Rotation for phase marking
            qc.z(qreg_ancilla[0])

            # Multi-controlled NOT undo
            if n_qubits == 1:
                qc.cx(qreg[0], qreg_ancilla[0])
            else:
                qc.mct(list(qreg), qreg_ancilla[0])

            # X-Gates undo
            for j, bit in enumerate(binary_repr):
                if bit == '0':
                    qc.x(qreg[j])
    
    return qc


def quantum_grover_minimum_search(distances: np.ndarray, target_count: int = 1, 
                                shots: int = 1024) -> np.ndarray:
    """
    Complete Quantum Grover Search for Minimum Finding.
    Finds the k smallest distances without classical comparisons.
    """
    n_items = len(distances)
    n_qubits = int(np.ceil(np.log2(n_items)))
    
    if n_items == 0:
        return np.array([])

    # Adaptive threshold for Grover Search
    sorted_distances = np.sort(distances)
    if target_count < len(sorted_distances):
        threshold = sorted_distances[target_count]
    else:
        threshold = np.max(distances) + 1
    
    # Quantum Registers
    qreg = QuantumRegister(n_qubits, 'search')
    qreg_ancilla = QuantumRegister(1, 'oracle')
    creg = ClassicalRegister(n_qubits, 'result')
    
    qc = QuantumCircuit(qreg, qreg_ancilla, creg)

    # 1. Superposition over all indices
    qc.h(qreg)

    # 2. Estimate number of marked states
    n_marked = np.sum(distances < threshold)
    if n_marked == 0:
        return np.arange(min(target_count, len(distances)))

    # 3. Optimal number of Grover iterations
    if n_marked > 0:
        optimal_iterations = int(np.pi/4 * np.sqrt(n_items/n_marked))
    else:
        optimal_iterations = 1

    # 4. Grover iterations
    for _ in range(optimal_iterations):
        # use Oracle 
        oracle = quantum_minimum_finding_oracle(distances, threshold)
        qc.compose(oracle, qubits=list(qreg) + [qreg_ancilla[0]], inplace=True)
        
        # Diffusion Operator (Amplitude Amplification)
        qc.h(qreg)
        qc.x(qreg)
        
        # Multi-controlled Z 
        if n_qubits == 1:
            qc.z(qreg[0])
        else:
            qc.mct(qreg[:-1], qreg[-1])
            qc.z(qreg[-1])
            qc.mct(qreg[:-1], qreg[-1])
        
        qc.x(qreg)
        qc.h(qreg)

    # 5. Measurement
    qc.measure(qreg, creg)
    
    # 6. real quantum execution
    backend = Aer.get_backend('qasm_simulator')
    transpiled_qc = transpile(qc, backend)
    job = execute(transpiled_qc, backend, shots=shots)
    result = job.result()
    counts = result.get_counts()

    # 7. Extract indices of most frequent measurements
    measured_indices = []
    for bitstring, count in counts.items():
        index = int(bitstring, 2)
        if index < len(distances):
            measured_indices.extend([index] * count)

    # 8. Return the most frequent indices
    if measured_indices:
        unique_indices, counts_per_index = np.unique(measured_indices, return_counts=True)
        sorted_indices = unique_indices[np.argsort(-counts_per_index)]
        return sorted_indices[:target_count]
    else:
        return np.arange(min(target_count, len(distances)))


def quantum_parallel_cluster_assignment(data_points: np.ndarray, centroids: np.ndarray, 
                                      shots: int = 1024) -> np.ndarray:
    """
    Full Quantum Parallel Cluster Assignment.
    Uses quantum superposition for parallel distance calculation and Grover search.
    """
    n_points = len(data_points)
    assignments = np.zeros(n_points, dtype=int)
    
    for point_idx, data_point in enumerate(data_points):
        # Calculate quantum distances to all centroids
        quantum_distances = quantum_distance_calculation_parallel(data_point, centroids, shots)

        # Find minimum with Grover Search
        min_indices = quantum_grover_minimum_search(quantum_distances, target_count=1, shots=shots)
        
        if len(min_indices) > 0:
            assignments[point_idx] = min_indices[0]
        else:
            assignments[point_idx] = 0
    
    return assignments


def quantum_distance_calculation_parallel(query_point: np.ndarray, 
                                        reference_points: np.ndarray,
                                        shots: int = 1024) -> np.ndarray:
    """
    Parallel quantum distance calculation to multiple reference points.
    Uses quantum superposition for simultaneous calculation of all distances.
    """
    n_refs = len(reference_points)
    distances = np.zeros(n_refs)

    # Normalization for Amplitude Encoding
    query_norm = query_point / (np.linalg.norm(query_point) + 1e-12)
    
    for i, ref_point in enumerate(reference_points):
        ref_norm = ref_point / (np.linalg.norm(ref_point) + 1e-12)

        # Quantum Swap Test for distance calculation
        distances[i] = quantum_swap_test_distance(query_norm, ref_norm, shots)
    
    return distances


def quantum_swap_test_distance(vec1: np.ndarray, vec2: np.ndarray, shots: int = 1024) -> float:
    """
    Real quantum swap test distance calculation.
    Computes 1 - |<vec1|vec2>|² via quantum interference.
    """
    d = len(vec1)
    n_qubits = int(np.ceil(np.log2(d))) if d > 1 else 1
    
    qreg1 = QuantumRegister(n_qubits, 'vec1')
    qreg2 = QuantumRegister(n_qubits, 'vec2')
    qreg_ancilla = QuantumRegister(1, 'ancilla')
    creg = ClassicalRegister(1, 'result')
    
    qc = QuantumCircuit(qreg1, qreg2, qreg_ancilla, creg)
    
    try:
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-12)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-12)
        
        circuit1 = amplitude_encoding(vec1_norm)
        qc.compose(circuit1, qreg1, inplace=True)
        
        circuit2 = amplitude_encoding(vec2_norm)
        qc.compose(circuit2, qreg2, inplace=True)
        
        qc.h(qreg_ancilla[0])
        
        for i in range(n_qubits):
            qc.cswap(qreg_ancilla[0], qreg1[i], qreg2[i])
        
        qc.h(qreg_ancilla[0])
        qc.measure(qreg_ancilla[0], creg[0])
        
        backend = Aer.get_backend('qasm_simulator')
        transpiled_qc = transpile(qc, backend)
        job = execute(transpiled_qc, backend, shots=shots)
        result = job.result()
        counts = result.get_counts()
        
        prob_0 = counts.get('0', 0) / shots
        overlap_squared = 2 * prob_0 - 1  
        overlap_squared = max(0.0, min(1.0, overlap_squared))  
        
        return 1.0 - overlap_squared
        
    except Exception as e:
        print(f"[QUANTUM ERROR] Swap test failed: {e}")
        # classical Fallback if quantum fails
        classical_overlap_squared = np.abs(np.dot(vec1_norm, vec2_norm))**2
        return max(0.0, 1.0 - classical_overlap_squared)


def quantum_superposition_centroid_update(data_points: np.ndarray, 
                                        cluster_labels: np.ndarray,
                                        n_clusters: int,
                                        shots: int = 1024) -> np.ndarray:
    """
    Quantum Superposition Centroid Update.
    calculates new centroids via superposition of all clusters.
    """
    n_features = data_points.shape[1]
    new_centroids = np.zeros((n_clusters, n_features))
    
    for cluster_id in range(n_clusters):
        cluster_points = data_points[cluster_labels == cluster_id]
        
        if len(cluster_points) == 0:
            new_centroids[cluster_id] = np.random.randn(n_features)
            continue
        
        if len(cluster_points) == 1:
            new_centroids[cluster_id] = cluster_points[0]
            continue
        
        new_centroids[cluster_id] = quantum_superposition_mean(cluster_points, shots)
    
    return new_centroids


def quantum_superposition_mean(points: np.ndarray, shots: int = 1024) -> np.ndarray:
    """
    Calculates mean via quantum superposition.
    """
    n_points, n_features = points.shape
    
    if n_points <= 1:
        return points[0] if n_points == 1 else np.zeros(n_features)
    
    n_qubits = int(np.ceil(np.log2(n_points)))
    
    qreg = QuantumRegister(n_qubits, 'points')
    qreg_data = QuantumRegister(n_features, 'data') 
    creg = ClassicalRegister(n_features, 'result')
    
    qc = QuantumCircuit(qreg, qreg_data, creg)
    
    qc.h(qreg)
    
    
    for i, point in enumerate(points):
        if i < 2**n_qubits:
            
            binary = format(i, f'0{n_qubits}b')
            
            
            for j, bit in enumerate(binary):
                if bit == '0':
                    qc.x(qreg[j])

            # Multi-Controlled Amplitude Encoding
            normalized_point = point / (np.linalg.norm(point) + 1e-12)
            for k, amplitude in enumerate(normalized_point):
                if k < n_features and np.abs(amplitude) > 1e-12:
                    angle = 2 * np.arcsin(min(1.0, np.abs(amplitude)))
                    
                    if n_qubits == 1:
                        qc.cry(angle, qreg[0], qreg_data[k])
                    else:
                        qc.mcry(angle, list(qreg), qreg_data[k])
            
            
            for j, bit in enumerate(binary):
                if bit == '0':
                    qc.x(qreg[j])
    
    qc.measure(qreg_data, creg)
    
    
    backend = Aer.get_backend('qasm_simulator')
    transpiled_qc = transpile(qc, backend, optimization_level=2)
    job = execute(transpiled_qc, backend, shots=shots)
    result = job.result()
    counts = result.get_counts()
    
    feature_expectations = np.zeros(n_features)
    
    for bitstring, count in counts.items():
        for i, bit in enumerate(bitstring[::-1]):
            if i < n_features:
                feature_expectations[i] += int(bit) * count / shots
    original_scale = np.mean([np.linalg.norm(point) for point in points])
    return feature_expectations * original_scale


def quantum_amplitude_estimation_distance(vec1: np.ndarray, vec2: np.ndarray,
                                        precision_qubits: int = 3,
                                        shots: int = 1024) -> float:
    
    vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-12)
    vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-12)
    
    d = len(vec1)
    n_qubits = int(np.ceil(np.log2(d))) if d > 1 else 1
    qc_state_prep = QuantumCircuit(2 * n_qubits + 1)
    
    circuit1 = amplitude_encoding(vec1_norm)
    circuit2 = amplitude_encoding(vec2_norm)
    
    qc_state_prep.compose(circuit1, range(n_qubits), inplace=True)
    qc_state_prep.compose(circuit2, range(n_qubits, 2*n_qubits), inplace=True)
    
    qc_state_prep.h(2*n_qubits)
    for i in range(n_qubits):
        qc_state_prep.cswap(2*n_qubits, i, n_qubits + i)

    backend = Aer.get_backend('qasm_simulator')

    amplitude_estimates = []
    
    for rotation_angle in np.linspace(0, np.pi, 8):
        qc_measure = qc_state_prep.copy()
        qc_measure.ry(rotation_angle, 2*n_qubits)
        qc_measure.measure_all()
        
        job = execute(qc_measure, backend, shots=shots//8)
        result = job.result()
        counts = result.get_counts()
        
        total_shots = shots//8
        prob_ancilla_0 = 0
        
        for bitstring, count in counts.items():
            # Check ancilla qubit 
            if len(bitstring) >= 2*n_qubits + 1:
                ancilla_bit = bitstring[-(2*n_qubits+1)]
                if ancilla_bit == '0':
                    prob_ancilla_0 += count
        
        prob_ancilla_0 /= total_shots
        
        overlap_squared = 2 * prob_ancilla_0 - 1
        amplitude_estimates.append(max(0, overlap_squared))
    
    estimated_overlap_squared = np.mean(amplitude_estimates)
    estimated_overlap = np.sqrt(max(0, estimated_overlap_squared))
    
    
    distance_squared = 2 * (1 - estimated_overlap)
    return np.sqrt(max(0, distance_squared))