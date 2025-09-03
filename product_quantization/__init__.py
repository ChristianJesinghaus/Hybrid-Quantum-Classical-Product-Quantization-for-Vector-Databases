# -*- coding: utf-8 -*-
__author__ = 'Christian Jesinghaus'

# © 2025 Christian Jesinghaus
# SPDX-License-Identifier: LicenseRef-BA-Citation


from .PQKNN import ProductQuantizationKNN
from .quantum_pqknn import QuantumProductQuantizationKNN
from .quantum_kmeans import QuantumKMeans
from .quantum_distance import QuantumDistanceCalculator, quantum_pairwise_distances
from .quantum_simulator import QuantumSimulator, QRAMSimulator
from .txt_config_loader import ConfigLoader
from .model_persistence import ModelPersistence
from .experiment_utils import generate_experiment_name, print_evaluation_summary
from .normalize import normalize_data, l2_normalize, quantum_amplitude_normalize


# Export main classes and functions
__all__ = [
    # Classical implementation
    'ProductQuantizationKNN',
    # Quantum implementations
    'QuantumProductQuantizationKNN',
    'QuantumKMeans',
    'QuantumDistanceCalculator',
    'QuantumSimulator',
    'QRAMSimulator',
    # Utility functions
    'quantum_pairwise_distances',
    # Configuration and persistence
    'ConfigLoader',
    'ModelPersistence',
    # Experiment utilities
    'generate_experiment_name',
    'print_evaluation_summary',
    # Normalization functions
    'normalize_data',
    'l2_normalize',
    'quantum_amplitude_normalize'
]