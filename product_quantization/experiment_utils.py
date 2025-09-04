# -*- coding: utf-8 -*-
__author__ = 'Christian Jesinghaus'
 
# © 2025 Christian Jesinghaus
# SPDX-License-Identifier: LicenseRef-BA-Citation

import os
import json
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

def generate_experiment_name(config):
    """
    Using config params for naming the model.
    """
    experiment_name = f"pqknn_n{config.get('n')}_c{config.get('c')}"

    algo = str(config.get('algorithm', 'quantum')).lower()
    if algo in ('quantum', 'classical'):
        experiment_name += f"_{algo}"
    else:
        experiment_name += "_unknown"

    quantum_shots = config.get('quantum_shots')
    if quantum_shots and algo == 'quantum':
        if quantum_shots >= 1000:
            experiment_name += f"_s{quantum_shots//1000}k"
        else:
            experiment_name += f"_s{quantum_shots}"

    return experiment_name

def print_evaluation_summary(preds, true_labels, verbose=True):
    """
    :param preds
    :param true_labels
    :param verbose: if true, then detailed report
    :return: Accuracy
    """
    accuracy = np.mean(preds == true_labels)
    print(f"[INFO] Total accuracy: {accuracy*100:.2f}%")
    
    if verbose:
        #detailed resulsts
        print("\nPredictions vs. True:")
        for i in range(min(10, len(preds))):  # show only the first 10 for spamming reasons
            print(f"Sample {i}: pred={preds[i]}, true={true_labels[i]}")
        
        if len(preds) > 10:
            print(f"... and {len(preds) - 10} more samples")
        
        try:
            report = classification_report(true_labels, preds)
            print("\nDetailed classification report:")
            print(report)
        except Exception as e:
            print(f"[WARN] Could not generate classification report: {e}")
    return accuracy

def save_experiment_results(results, filename):
    """
    Save experiment results to file.
    """    
    try:
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"[INFO] Results saved to {filename}")
    except Exception as e:
        print(f"[ERROR] Failed to save results: {e}")

def load_experiment_results(filename):
    """
    Load experiment results from file.
    """    
    try:
        with open(filename, 'r') as f:
            results = json.load(f)
        print(f"[INFO] Results loaded from {filename}")
        return results
    except Exception as e:
        print(f"[ERROR] Failed to load results: {e}")
        return None