# -*- coding: utf-8 -*-
__author__ = 'Christian Jesinghaus'

# © 2025 Christian Jesinghaus
# SPDX-License-Identifier: LicenseRef-BA-Citation

import numpy as np
from typing import Union


def normalize_data(data: np.ndarray, method: str = 'l2') -> np.ndarray:
    """
        Normalization function offering different normalization methods  
    Args:
        data: Input data array (samples x features)
        method: Normalization method ('l2', 'minmax', 'standard', 'unit')
        
    Returns:
        Normalized data array
    """
    if method == 'l2':
        return l2_normalize(data)
    elif method == 'minmax':
        return minmax_normalize(data)
    elif method == 'standard':
        return standard_normalize(data)
    elif method == 'unit':
        return unit_normalize(data)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def l2_normalize(data: np.ndarray) -> np.ndarray:
    """
    L2 normalization - each sample has unit L2 norm.
    
    Args:
        data: Input data array (samples x features)
        
    Returns:
        L2 normalized data
    """
    normalized_data = np.copy(data).astype(np.float64)
    
    for i in range(len(normalized_data)):
        norm = np.linalg.norm(normalized_data[i])
        if norm > 1e-9:  # Avoid division by zero
            normalized_data[i] = normalized_data[i] / norm
        else:
            # If norm is zero, keep the vector as is (all zeros)
            pass
    
    return normalized_data


def minmax_normalize(data: np.ndarray) -> np.ndarray:
    """
    Min-Max normalization - scale features to [0, 1] range.
    
    Args:
        data: Input data array (samples x features)
        
    Returns:
        Min-Max normalized data
    """
    normalized_data = np.copy(data).astype(np.float64)
    
    # Normalize each feature independently
    for j in range(data.shape[1]):
        feature_min = np.min(data[:, j])
        feature_max = np.max(data[:, j])
        
        if feature_max - feature_min > 1e-9:  # Avoid division by zero
            normalized_data[:, j] = (data[:, j] - feature_min) / (feature_max - feature_min)
        else:
            # If all values are the same, set to 0
            normalized_data[:, j] = 0.0
    
    return normalized_data


def standard_normalize(data: np.ndarray) -> np.ndarray:
    """
    Standard normalization (z-score) - zero mean, unit variance.
    
    Args:
        data: Input data array (samples x features)
        
    Returns:
        Standard normalized data
    """
    normalized_data = np.copy(data).astype(np.float64)
    
    # Normalize each feature independently
    for j in range(data.shape[1]):
        feature_mean = np.mean(data[:, j])
        feature_std = np.std(data[:, j])
        
        if feature_std > 1e-9:  # Avoid division by zero
            normalized_data[:, j] = (data[:, j] - feature_mean) / feature_std
        else:
            # If std is zero, center around zero
            normalized_data[:, j] = data[:, j] - feature_mean
    
    return normalized_data


def unit_normalize(data: np.ndarray) -> np.ndarray:
    """
    Unit normalization - scale each feature to unit variance.
    
    Args:
        data: Input data array (samples x features)
        
    Returns:
        Unit normalized data
    """
    normalized_data = np.copy(data).astype(np.float64)
    
    # Normalize each feature to unit variance
    for j in range(data.shape[1]):
        feature_std = np.std(data[:, j])
        
        if feature_std > 1e-9:  # Avoid division by zero
            normalized_data[:, j] = data[:, j] / feature_std
    
    return normalized_data


def robust_normalize(data: np.ndarray) -> np.ndarray:
    """
    Robust normalization using median and IQR.
    Less sensitive to outliers than standard normalization.
    
    Args:
        data: Input data array (samples x features)
        
    Returns:
        Robust normalized data
    """
    normalized_data = np.copy(data).astype(np.float64)
    
    # Normalize each feature independently
    for j in range(data.shape[1]):
        feature_median = np.median(data[:, j])
        q75, q25 = np.percentile(data[:, j], [75, 25])
        iqr = q75 - q25
        
        if iqr > 1e-9:  # Avoid division by zero
            normalized_data[:, j] = (data[:, j] - feature_median) / iqr
        else:
            # If IQR is zero, center around median
            normalized_data[:, j] = data[:, j] - feature_median
    
    return normalized_data


def quantum_amplitude_normalize(data: np.ndarray) -> np.ndarray:
    """
    Normalize data for quantum amplitude encoding.
    Ensures each vector has unit L2 norm for proper quantum state preparation.
    
    Args:
        data: Input data array (samples x features)
        
    Returns:
        Amplitude-normalized data suitable for quantum encoding
    """
    normalized_data = np.copy(data).astype(np.float64)
    
    for i in range(len(normalized_data)):
        # L2 normalization for quantum amplitude encoding
        norm = np.linalg.norm(normalized_data[i])
        
        if norm > 1e-9:
            normalized_data[i] = normalized_data[i] / norm
        else:
            # For zero vectors, create a uniform distribution
            normalized_data[i] = np.ones(data.shape[1]) / np.sqrt(data.shape[1])
    
    return normalized_data


def normalize_for_quantum_distance(data: np.ndarray) -> np.ndarray:
    """
    Special normalization for quantum distance calculations.
    Combines min-max scaling with amplitude normalization.
    
    Args:
        data: Input data array (samples x features)
        
    Returns:
        Quantum-distance normalized data
    """
    # First apply min-max normalization to bring all features to [0,1]
    minmax_data = minmax_normalize(data)
    
    # Then apply amplitude normalization for quantum encoding
    quantum_data = quantum_amplitude_normalize(minmax_data)
    
    return quantum_data


def check_normalization(data: np.ndarray, method: str = 'l2') -> dict:
    """
    Check if data is properly normalized according to the specified method.
    
    Args:
        data: Data to check
        method: Normalization method to check against
        
    Returns:
        Dictionary with normalization statistics
    """
    stats = {}
    
    if method == 'l2':
        norms = [np.linalg.norm(row) for row in data]
        stats['mean_norm'] = np.mean(norms)
        stats['std_norm'] = np.std(norms)
        stats['min_norm'] = np.min(norms)
        stats['max_norm'] = np.max(norms)
        stats['is_normalized'] = np.allclose(norms, 1.0, atol=1e-6)
        
    elif method == 'minmax':
        stats['min_value'] = np.min(data)
        stats['max_value'] = np.max(data)
        stats['is_normalized'] = (stats['min_value'] >= -1e-6 and stats['max_value'] <= 1 + 1e-6)
        
    elif method == 'standard':
        feature_means = np.mean(data, axis=0)
        feature_stds = np.std(data, axis=0)
        stats['mean_of_means'] = np.mean(feature_means)
        stats['mean_of_stds'] = np.mean(feature_stds)
        stats['is_normalized'] = (np.allclose(feature_means, 0.0, atol=1e-6) and 
                                np.allclose(feature_stds, 1.0, atol=1e-6))
    
    return stats


# Export all functions
__all__ = [
    'normalize_data',
    'l2_normalize', 
    'minmax_normalize',
    'standard_normalize',
    'unit_normalize',
    'robust_normalize',
    'quantum_amplitude_normalize',
    'normalize_for_quantum_distance',
    'check_normalization'
]