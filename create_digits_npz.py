# © 2025 Christian Jesinghaus
# SPDX-License-Identifier: LicenseRef-BA-Citation 

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def create_digits_npz(file_out='example_data.npz',
                     train_size=1000,  # Beispiel: sehr wenige zum schnellen Testen
                     test_size=200):
    """
    
    Loads the Digits dataset from scikit-learn, reduces it to (train_size + test_size) samples
    and saves the arrays to a .npz file.
    This makes it practical for a hybrid PQ-KNN (Quantum-KMeans)
    and significantly faster than the large MNIST dataset.

    :param file_out: Filename for the output (e.g. 'example_data.npz')
    :param train_size: Number of training samples to use
    :param test_size: Number of test samples to use
    """
    print("[INFO] Loading Digits dataset from scikit-learn...")
    digits = load_digits()  # 1797 samples with 64 dimensions (8x8 image)

    data_all = digits.data
    labels_all = digits.target

    # Example: 80/20 split. You can adjust the split factor.
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        data_all, labels_all, test_size=0.2, random_state=42
    )

    # Limit to 'train_size' / 'test_size' samples
    X_train = X_train_full[:train_size]
    y_train = y_train_full[:train_size]
    X_test = X_test_full[:test_size]
    y_test = y_test_full[:test_size]

    print(f"[INFO] Using {len(X_train)} training samples, {len(X_test)} test samples.")

    # Save to npz file
    print(f"[INFO] Saving to {file_out} ...")
    np.savez(file_out,
             train_data=X_train,
             train_labels=y_train,
             test_data=X_test,
             test_labels=y_test)
    print("[INFO] Done.")


if __name__ == "__main__":
    # You can adjust the default values here:
    create_digits_npz(file_out='example_data.npz',
                     train_size=1000, 
                     test_size=200) 