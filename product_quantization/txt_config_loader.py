# -*- coding: utf-8 -*-
__author__ = 'Christian Jesinghaus'
 
# © 2025 Christian Jesinghaus
# SPDX-License-Identifier: LicenseRef-BA-Citation



import os
from typing import Dict, Any, Union


class ConfigLoader:
    """
    Configuration loader for hybrid quantum PQKNN experiments.
    Loads configuration from txt files with key-value pairs.
    """
    
    def __init__(self, config_file: str = "config.txt"):
        """
        Initialize config loader.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file
        self.config = {}
        
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Returns:
            Dictionary with configuration parameters
        """
        if not os.path.exists(self.config_file):
            print(f"[WARN] Config file {self.config_file} not found. Using defaults.")
            return self._get_default_config()
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse key-value pairs
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Convert value to appropriate type
                        self.config[key] = self._parse_value(value)
                    else:
                        print(f"[WARN] Invalid config line {line_num}: {line}")
            
            # Merge with defaults for missing keys
            default_config = self._get_default_config()
            for key, default_value in default_config.items():
                if key not in self.config:
                    self.config[key] = default_value
            
            print(f"[INFO] Loaded config from {self.config_file}")
            return self.config
            
        except Exception as e:
            print(f"[ERROR] Failed to load config: {e}")
            return self._get_default_config()
    
    def _parse_value(self, value: str) -> Union[int, float, bool, str]:
        """
        Parse string value to appropriate Python type.
        """
        # Remove quotes if present
        if (value.startswith('"') and value.endswith('"')) or \
           (value.startswith("'") and value.endswith("'")):
            return value[1:-1]
        
        # Boolean values
        if value.lower() in ['true', 'yes', '1']:
            return True
        elif value.lower() in ['false', 'no', '0']:
            return False
        
        # None value
        if value.lower() in ['none', 'null']:
            return None
        
        # Numeric values
        try:
            if '.' in value or 'e' in value.lower():
                return float(value)
            else:
                return int(value)
        except ValueError:
            # Return as string if can't convert
            return value
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration values.
        """
        return {
            # Data parameters
            'train_size': 1000,
            'test_size': 200,
            'data_file': 'example_data.npz',
            'normalize_data': True,

            # Log-Fidelity parameters 
            'use_log_fidelity': True,
            'log_fidelity_precision': 1e-9,
            
            # PQKNN algorithm parameters
            'n': 4,  # Number of partitions
            'c': 8,  # Number of clusters per partition
            'k': 5,  # Number of nearest neighbors
            

            'algorithm': 'quantum',  # 'classical' | 'quantum'
            # Quantum parameters
            'quantum_shots': 1024,
            'max_iter_qk': 15,
            
            # Experiment settings
            'random_state': 42,
            'verbose': True,
            'save_model': True,
            'model_output_dir': 'experiments/models',
            'create_experiment_report': True,
            
            
            
          
        }
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration parameters.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if valid, False otherwise
        """
        valid = True
        
        # Validate required parameters
        required_params = ['n', 'c', 'k', 'train_size', 'test_size']
        for param in required_params:
            if param not in config:
                print(f"[ERROR] Missing required parameter: {param}")
                valid = False
            elif not isinstance(config[param], int) or config[param] <= 0:
                print(f"[ERROR] Parameter {param} must be positive integer")
                valid = False
        
        return valid