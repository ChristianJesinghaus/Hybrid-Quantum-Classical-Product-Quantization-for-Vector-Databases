
# © 2025 Christian Jesinghaus
# SPDX-License-Identifier: LicenseRef-BA-Citation


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_confusion_matrix(true_labels, predictions, classes=None, title='Confusion Matrix'):
    """
    Creates a visualization of the confusion matrix.

    :param true_labels: True labels
    :param predictions: Predictions
    :param classes: Class names (optional)
    :param title: Title of the plot
    """
    cm = confusion_matrix(true_labels, predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.tight_layout()
    
    return plt.gcf()

def plot_performance_comparison(accuracy_dict, title='Performance Comparison'):
    """
    Creates a bar chart to compare different model configurations.

    :param accuracy_dict: Dictionary with configuration name -> Accuracy
    :param title: Title of the plot
    """
    configs = list(accuracy_dict.keys())
    accuracies = [accuracy_dict[c] for c in configs]
    
    plt.figure(figsize=(12, 6))
    plt.bar(configs, accuracies, color='skyblue')
    plt.axhline(y=max(accuracies), color='r', linestyle='-', alpha=0.3)
    plt.ylim(0, 1.0)
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return plt.gcf()

def plot_qkm_objective(history_list, title='Quantum KMeans Objective'):
    """
    history_list: List of partition histories (each a list of iteration dicts)
    Plots the mean (and optional band) of 'objective_after' over iterations
    """
    import matplotlib.pyplot as plt
    import numpy as np
    # auf gemeinsame Iterationsanzahl trimmen
    L = min(len(h) for h in history_list if h)
    if L == 0: return None
    mat = np.array([[h[i]['objective_after'] for i in range(L)] for h in history_list if h])
    mean = mat.mean(axis=0); std = mat.std(axis=0)
    x = np.arange(1, L+1)
    plt.figure(figsize=(7,4))
    plt.plot(x, mean)
    plt.fill_between(x, mean-std, mean+std, alpha=0.2)
    plt.xlabel('Iteration'); plt.ylabel('Objective')
    plt.title(title); plt.tight_layout()
    return plt.gcf()
