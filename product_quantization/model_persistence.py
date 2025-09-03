# -*- coding: utf-8 -*-
__author__ = 'Christian Jesinghaus'

# © 2025 Christian Jesinghaus
# SPDX-License-Identifier: LicenseRef-BA-Citation


import os
import pickle
import json
import shutil
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
from .PQKNN import ProductQuantizationKNN

from .quantum_pqknn import QuantumProductQuantizationKNN
from .experiment_utils import generate_experiment_name


class ModelPersistence:
    def __init__(self, base_dir: str = "experiments/models"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def save_model(self, model,  
                   config: Dict[str, Any],
                   experiment_results: Dict[str, Any],
                   model_name: Optional[str] = None) -> str:

        algo = (config.get('algorithm') or
                ('quantum' if isinstance(model, QuantumProductQuantizationKNN) else 'classical'))

        if model_name is None:
            model_name = generate_experiment_name(config)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(self.base_dir, f"{model_name}_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)

        try:
            #  MODEL STATE
            model_path = os.path.join(model_dir, "model.pkl")
            state = {
                'compressed_data': getattr(model, 'compressed_data', None),
                'subvector_centroids': getattr(model, 'subvector_centroids', {}),
                'train_labels': getattr(model, 'train_labels', None),
                'partition_size': getattr(model, 'partition_size', None),
                'd': getattr(model, 'd', None),
                'n': getattr(model, 'n', None),
                'c': getattr(model, 'c', None),
                'k': getattr(model, 'k', None),
                'quantum_shots': getattr(model, 'quantum_shots', None),
                'random_state': getattr(model, 'random_state', None),
                'distance_metric': getattr(model, 'distance_metric', None),
                'smooth_eps': getattr(model, 'smooth_eps', None),
            }
            with open(model_path, 'wb') as f:
                pickle.dump(state, f)

            # safe config
            config_out = dict(config)
            config_out['algorithm'] = algo  
            config_path = os.path.join(model_dir, "config.json")
            with open(config_path, 'w') as f:
                json.dump(config_out, f, indent=2, default=str)

            # safe results
            results_path = os.path.join(model_dir, "results.json")
            with open(results_path, 'w') as f:
                json.dump(experiment_results, f, indent=2, default=str)

            #quantum info
            qi = {}
            if hasattr(model, 'get_quantum_info'):
                try:
                    qi = model.get_quantum_info() or {}
                except Exception:
                    qi = {}
            quantum_path = os.path.join(model_dir, "quantum_info.json")
            with open(quantum_path, 'w') as f:
                json.dump(qi, f, indent=2, default=str)

            # compression stats
            total_centroids = 0
            for cents in getattr(model, 'subvector_centroids', {}).values():
                try:
                    total_centroids += len(cents)
                except Exception:
                    pass

            comp_ratio = None
            if hasattr(model, 'get_compression_ratio'):
                try:
                    comp_ratio = model.get_compression_ratio()
                except Exception:
                    comp_ratio = None

            stats = {
                'compression_ratio': comp_ratio,
                'compressed_data_shape': (getattr(model, 'compressed_data', None).shape
                                          if getattr(model, 'compressed_data', None) is not None else None),
                'original_dimensions': getattr(model, 'd', None),
                'partitions': getattr(model, 'n', None),
                'clusters_per_partition': getattr(model, 'c', None) or getattr(model, 'k', None),
                'total_centroids': total_centroids
            }
            stats_path = os.path.join(model_dir, "compression_stats.json")
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2, default=str)

            # safe histories
            hist_attr = None
            if hasattr(model, "kmeans_histories"):
                hist_attr = "kmeans_histories"
            elif hasattr(model, "subvector_histories"):
                hist_attr = "subvector_histories"
            if hist_attr:
                try:
                    histories = getattr(model, hist_attr, {})
                    hist_dir = os.path.join(model_dir, "histories")
                    os.makedirs(hist_dir, exist_ok=True)
                    for p_idx, metrics in histories.items():
                        if not metrics:
                            continue
                        if hist_attr == "kmeans_histories":
                            filename = f"partition_{p_idx:02d}_kmeans_stats.json"
                        else:
                            filename = f"partition_{p_idx:02d}_qkm_history.json"
                        path = os.path.join(hist_dir, filename)
                        with open(path, "w", encoding="utf-8") as fh:
                            json.dump(metrics, fh, indent=2, ensure_ascii=False)
                except Exception:
                    pass

            self._create_model_summary(model_dir, model_name, config_out, experiment_results)

            print(f"[INFO] Model saved to {model_dir}")
            return model_dir

        except Exception as e:
            print(f"[ERROR] Failed to save model: {e}")
            return None

    def load_model(self, model_path: str) -> tuple:
        try:
            if os.path.isfile(model_path):
                model_dir = os.path.dirname(model_path)
                model_file = model_path
            else:
                model_dir = model_path
                model_file = os.path.join(model_dir, "model.pkl")

            with open(model_file, 'rb') as f:
                model_state = pickle.load(f)

            #load config and results
            config_path = os.path.join(model_dir, "config.json")
            config = {}
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)

            results_path = os.path.join(model_dir, "results.json")
            results = {}
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    results = json.load(f)
            algo = (config.get('algorithm') or
                    ('quantum' if (model_state.get('quantum_shots') is not None or
                                   model_state.get('distance_metric') is not None) else 'classical'))

            # reconstruct model
            if algo == 'classical':
                model = ProductQuantizationKNN(
                    n=model_state.get('n'),
                    c=0,  
                    k_clusters=model_state.get('c') or model_state.get('k'),
                    random_state=model_state.get('random_state', 42)
                )
            else:
                model = QuantumProductQuantizationKNN(
                    n=model_state.get('n'),
                    c=model_state.get('c') or model_state.get('k'),
                    max_iter_qk=config.get('max_iter_qk', 15),
                    quantum_shots=model_state.get('quantum_shots', 1024),
                    random_state=model_state.get('random_state', 42),
                    distance_metric=model_state.get('distance_metric', 'log_fidelity'),
                    smooth_eps=model_state.get('smooth_eps', 1e-3)
                )
            model.compressed_data = model_state.get('compressed_data')
            model.subvector_centroids = model_state.get('subvector_centroids', {})
            model.train_labels = model_state.get('train_labels')
            model.partition_size = model_state.get('partition_size')
            model.d = model_state.get('d')

            print(f"[INFO] Model loaded from {model_dir}")
            return model, config, results

        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            return None, None, None

    def _create_model_summary(self, model_dir: str, model_name: str,
                              config: Dict[str, Any], results: Dict[str, Any]):
        summary_path = os.path.join(model_dir, "model_summary.txt")
        try:
            with open(summary_path, 'w') as f:
                f.write("="*60 + "\n")
                f.write(f"HYBRID/CLASSICAL PQKNN MODEL SUMMARY\n")
                f.write("="*60 + "\n\n")

                f.write(f"Model Name: {model_name}\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                f.write("CONFIGURATION:\n")
                f.write("-" * 30 + "\n")
                for key, value in config.items():
                    f.write(f"{key:25}: {value}\n")
                f.write("\n")

                f.write("RESULTS:\n")
                f.write("-" * 30 + "\n")
                for key, value in results.items():
                    if isinstance(value, float):
                        f.write(f"{key:25}: {value:.4f}\n")
                    else:
                        f.write(f"{key:25}: {value}\n")
                f.write("\n")

                algo = str(config.get('algorithm', 'unknown')).lower()
                f.write("ALGORITHM INFO:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Algorithm: {algo}\n")
                if algo == 'quantum':
                    f.write(f"Quantum Shots: {config.get('quantum_shots', 'N/A')}\n")
                    f.write(f"Distance Metric: {config.get('distance_metric', 'N/A')}\n")
                    f.write(f"QRAM Simulation: Yes\n")

        except Exception as e:
            print(f"[WARN] Could not create model summary: {e}")

   
    def list_saved_models(self) -> list:
        """
        List all saved models in the base directory.
        Returns:
            List of model directory names
        """
        try:
            models = []
            for item in os.listdir(self.base_dir):
                item_path = os.path.join(self.base_dir, item)
                if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "model.pkl")):
                    models.append(item)
            return sorted(models)
        except Exception as e:
            print(f"[ERROR] Failed to list models: {e}")
            return []
    
    def delete_model(self, model_name: str) -> bool:
        """
        Delete a saved model.
        Args:
            model_name: Name of model directory to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            model_path = os.path.join(self.base_dir, model_name)
            if os.path.exists(model_path):
                shutil.rmtree(model_path)
                print(f"[INFO] Model {model_name} deleted")
                return True
            else:
                print(f"[WARN] Model {model_name} not found")
                return False
        except Exception as e:
            print(f"[ERROR] Failed to delete model: {e}")
            return False
    
    def export_model_for_deployment(self, model_path: str, output_path: str):
        """
        Export model in a format suitable for deployment.
        
        Args:
            model_path: Path to saved model
            output_path: Path for deployment package
        """
        try:
            model, config, results = self.load_model(model_path)
            if model is None:
                return False
            
            # Create minimal deployment package
            deployment_data = {
                'model_type': 'QuantumProductQuantizationKNN',
                'compressed_data': model.compressed_data,
                'subvector_centroids': model.subvector_centroids,
                'train_labels': model.train_labels,
                'model_params': {
                    'n': model.n,
                    'c': model.c,
                    'partition_size': model.partition_size,
                    'd': model.d
                },
                'prediction_config': {
                    'quantum_shots': config.get('quantum_shots', 1024)                }
            }
            
            with open(output_path, 'wb') as f:
                pickle.dump(deployment_data, f)
            
            print(f"[INFO] Model exported for deployment to {output_path}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to export model: {e}")
            return False