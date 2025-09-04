# -*- coding: utf-8 -*-
__author__ = 'Jeroen Van Der Donckt'
 
# © 2021 Jeroen Van Der Donckt — License: MIT (see THIRD_PARTY_LICENSES/classical_PQKNN-MIT.txt)
# Modifications © 2025 Christian Jesinghaus
# SPDX-License-Identifier: MIT


import multiprocessing
from typing import Callable, Any
import os
import json
import numpy as np
from sklearn.cluster import KMeans

from .util import log_nb_clusters_to_np_int_type, squared_euclidean_dist


class ProductQuantizationKNN:
    """
    Classical Product‑Quantization kNN implementation.

    Parameters
    ----------
    n : int
        Number of partitions (subvectors) into which each sample is split.
    c : int, optional
        Logarithm base‑2 of the number of clusters per partition.  Used if
        ``k_clusters`` is not provided; the effective number of clusters
        becomes ``2**c``.
    k_clusters : int, optional
        Exact number of clusters per partition.  If provided, overrides
        ``c`` and allows arbitrary cluster counts (not necessarily a power
        of two).
    random_state : int, optional
        Random seed forwarded to scikit‑learn's `KMeans`.  Defaults to ``42``
        to maintain backwards compatibility.
    n_init : int, optional
        Number of K‑Means initializations to run.  See scikit‑learn's
        documentation for details.  Defaults to ``1`` to mirror the original
        behaviour.

    Notes
    -----
    * When using ``k_clusters``, the internal integer type is chosen based on
      ``ceil(log2(k_clusters))`` to ensure sufficient bits are used to index
      all clusters.
    * The `compress` method automatically attaches the remainder of the
      feature dimension to the final partition if the dimension is not
      divisible by ``n``.
    * Per‑partition K‑Means statistics (inertia, number of iterations,
      cluster sizes and fit time) are recorded and can be exported via
      :meth:`export_histories`.
    """

    def __init__(self,
                 n: int,
                 c: int | None = None,
                 *,
                 k_clusters: int | None = None,
                 random_state: int = 42,
                 n_init: int = 1) -> None:
        # number of partitions
        self.n: int = n
        # determine number of clusters
        if k_clusters is not None and k_clusters > 0:
            self.k = int(k_clusters)
            # derive c for backwards compatibility (log2 rounding up)
            # so that compression_stats.json still includes a meaningful 'c'
            import math
            self.c = int(math.ceil(math.log2(max(1, self.k))))
        elif c is not None:
            # original behaviour: k = 2**c
            self.c = int(c)
            self.k = 2 ** self.c
        else:
            raise ValueError("Either c or k_clusters must be provided")
        # choose minimal numpy integer type based on number of clusters
        import math
        bits = int(math.ceil(math.log2(max(1, self.k))))
        self.int_type = log_nb_clusters_to_np_int_type(bits)
        # hyperparameters for reproducible K‑Means
        self.random_state: int = int(random_state)
        self.n_init: int = int(n_init)
        # store centroids per partition after compression
        self.subvector_centroids: dict[int, np.ndarray] = {}
        # storage for K‑Means stats per partition
        # each entry will be a dict with keys: inertia, n_iter, cluster_sizes, fit_time_sec
        self.kmeans_histories: dict[int, dict[str, Any]] = {}
        # fields initialised during compression
        self.partition_size: int | None = None
        self.train_labels: np.ndarray | None = None
        self.compressed_data: np.ndarray | None = None
        self.d: int | None = None

    def _get_data_partition(self, train_data, partition_idx):
        partition_start = partition_idx * self.partition_size
        partition_end = (partition_idx + 1) * self.partition_size
        train_data_partition = train_data[:, partition_start:partition_end]
        return train_data_partition

    def _compress_partition(self, partition_idx: int, train_data_partition):
        """
        Helper function for compressing a data partition.

        Returns:
        - partition_idx      : Index of the partition
        - labels             : Cluster assignment per data point (dtype = int_type)
        - centroids          : Centroid matrix of this partition
        - metrics            : Dict with metrics (Inertia, n_iter, cluster_sizes,
                               fit_time_sec, inertia_history)
        """
        import time
        # Start timing
        t0 = time.perf_counter()
        # Main K-Means run (returns final centroids and metrics)
        km = KMeans(
            n_clusters=self.k,
            n_init=self.n_init,
            random_state=self.random_state,
        )
        labels = km.fit_predict(train_data_partition).astype(self.int_type)
        centroids = km.cluster_centers_
        inertia_final = float(getattr(km, "inertia_", np.nan))
        n_iter = int(getattr(km, "n_iter_", 0))
        # Determine cluster sizes (length = k)
        try:
            counts = np.bincount(labels, minlength=self.k).astype(int)
            cluster_sizes = counts.tolist()
        except Exception:
            cluster_sizes = []
        # Approximate inertia history: one-step K-Means runs
        inertia_history: list[float] = []
        centers_step = None
        for i in range(max(1, n_iter)):
            step_km = KMeans(
                n_clusters=self.k,
                init=("k-means++" if centers_step is None else centers_step),
                n_init=1,
                max_iter=1,
                random_state=self.random_state,
            )
            step_km.fit(train_data_partition)
            inertia_history.append(float(step_km.inertia_))
            centers_step = step_km.cluster_centers_
        #   Stop timing
        fit_time = float(time.perf_counter() - t0)
        metrics = {
            "inertia": inertia_final,
            "n_iter": n_iter,
            "cluster_sizes": cluster_sizes,
            "fit_time_sec": fit_time,
            "inertia_history": inertia_history,
        }
        return partition_idx, labels, centroids, metrics

    def compress(self, train_data: np.ndarray, train_labels: np.ndarray):
        """
        Compress the given training data via the product quantization method.

        This method splits each input vector into ``n`` partitions.  If the
        dimensionality is not divisible by ``n``, the remaining features are
        assigned to the last partition.  For each partition a separate K‑Means
        clustering with ``self.k`` clusters is trained using the provided
        ``random_state`` and ``n_init``.  The resulting cluster indices are
        stored in ``self.compressed_data`` and the centroids in
        ``self.subvector_centroids``.  Additionally, per‑partition K‑Means
        statistics are recorded in ``self.kmeans_histories``.

        Parameters
        ----------
        train_data : ndarray of shape (n_samples, n_features)
            Training examples.
        train_labels : ndarray of shape (n_samples,)
            Corresponding labels.  Must match the length of ``train_data``.
        """
        nb_samples = len(train_data)
        if nb_samples != len(train_labels):
            raise AssertionError(
                "The number of train samples does not match the length of the labels"
            )
        self.train_labels = np.asarray(train_labels)
        # initialise compressed data array
        self.compressed_data = np.empty((nb_samples, self.n), dtype=self.int_type)
        # total dimension and base partition size
        self.d = int(train_data.shape[1])
        self.partition_size = self.d // self.n if self.n > 0 else 0

        # prepare parameters for each partition, taking care of the remainder
        params: list[tuple[int, np.ndarray]] = []
        for p_idx in range(self.n):
            if p_idx == self.n - 1:
                # last partition takes the rest (may be equal in size if divisible)
                part = train_data[:, p_idx * self.partition_size :]
            else:
                part = self._get_data_partition(train_data, p_idx)
            params.append((p_idx, part))
        # run clustering (parallelised via multiprocessing)
        if len(params) > 0:
            with multiprocessing.Pool() as pool:
                res = pool.starmap(self._compress_partition, params)
            for (p_idx, labels, centroids, metrics) in res:
                # assign compressed codes and centroids
                self.compressed_data[:, p_idx] = labels
                self.subvector_centroids[p_idx] = centroids
                # store metrics
                self.kmeans_histories[p_idx] = metrics

    def predict_single_sample(self, test_sample: np.ndarray, nearest_neighbors: int,
                              calc_dist: Callable[[np.ndarray, np.ndarray], np.ndarray] = squared_euclidean_dist):
        """ Predicts the label of the given test sample based on the PQKNN algorithm

        :param test_sample: the test sample (a 1D array).
        :param nearest_neighbors: the k in kNN.
        :param calc_dist: the distance function that should be used. Defaults to squared euclidean distance.
        :return: the predicted label.
        """
        assert hasattr(self, 'compressed_data') and hasattr(self, 'train_labels'), \
            'There is no stored compressed data, therefore PQKNN can not do a k-NN search'
        # Compute table containing the distances of the sample to the centroids of each partition
        distances = np.empty(shape=(self.k, self.n), dtype=np.float64)
        for partition_idx in range(self.n):
            # determine slice for this partition
            partition_start = partition_idx * self.partition_size
            if partition_idx == self.n - 1:
                # last partition includes remainder of the vector
                test_sample_partition = test_sample[partition_start:]
            else:
                partition_end = (partition_idx + 1) * self.partition_size
                test_sample_partition = test_sample[partition_start:partition_end]
            centroids_partition = self.subvector_centroids[partition_idx]
            distances[:, partition_idx] = calc_dist(test_sample_partition, centroids_partition)

        # Calculate (approximate) distance to stored data
        nb_stored_samples = len(self.compressed_data)
        distance_sums = np.zeros(shape=nb_stored_samples)
        for partition_idx in range(self.n):
            distance_sums += distances[:, partition_idx][self.compressed_data[:, partition_idx]]

        # Select label among k nearest neighbors
        indices = np.argpartition(distance_sums, nearest_neighbors)
        labels = self.train_labels[indices][:nearest_neighbors]
        unique_labels, counts = np.unique(labels, return_counts=True)
        if len(unique_labels) == 1:
            return unique_labels[0]
        sorted_idxs = np.argsort(counts)[::-1]
        unique_labels = unique_labels[sorted_idxs]
        counts = counts[sorted_idxs]
        if counts[0] != counts[1]:
            return unique_labels[0]
        max_count = counts[0]
        idx = 0
        min_distance = float('inf')
        selected_label = None
        while idx < len(unique_labels) and counts[idx] == max_count:
            label = unique_labels[idx]
            label_indices = np.where(labels == label)
            summed_distance = np.sum(distance_sums[indices[label_indices]])
            if summed_distance < min_distance:
                selected_label = label
                min_distance = summed_distance
            idx += 1
        return selected_label

    def predict(self,
                test_data: np.ndarray,
                nearest_neighbors: int | None = None,
                calc_dist: Callable[[np.ndarray, np.ndarray], np.ndarray] = squared_euclidean_dist,
                **kwargs) -> np.ndarray:
        """
        Predict labels for a batch of test samples based on the PQKNN algorithm.

        Parameters
        ----------
        test_data : ndarray of shape (n_samples, n_features)
            The data points for which to predict labels.
        nearest_neighbors : int, optional
            The number of neighbours to consider when voting on the label.  If omitted, the
            value of the keyword argument ``k`` will be used instead.  Either ``nearest_neighbors``
            or ``k`` must be provided, otherwise a ``ValueError`` is raised.
        calc_dist : callable, optional
            Distance function used to compare a test subvector against the partition centroids.
            Defaults to the squared Euclidean distance.
        **kwargs : dict
            Additional keyword arguments are accepted for backwards compatibility.
            In particular, the keyword ``k`` is treated as an alias for ``nearest_neighbors`` to
            mirror the signature used in some example scripts.  If both are provided, the explicit
            ``k`` value takes precedence.

        Returns
        -------
        np.ndarray
            Array of predicted labels for each test sample.
        """
        # support 'k' as alias for nearest_neighbors; extract from kwargs
        k_alias = kwargs.get('k', None)
        # Determine the effective number of neighbours
        if nearest_neighbors is None:
            if k_alias is None:
                raise ValueError("Either nearest_neighbors or 'k' keyword argument must be provided")
            try:
                nearest_neighbors = int(k_alias)
            except Exception:
                raise ValueError(f"Invalid k value: {k_alias}")
        else:
            # if both provided, favour the explicit 'k' keyword
            if k_alias is not None:
                try:
                    nearest_neighbors = int(k_alias)
                except Exception:
                    pass
        # Validate test_data dimensionality
        assert test_data.ndim == 2, 'The dimensionality of the test_data should be 2'
        # ensure a positive number of neighbours
        if nearest_neighbors <= 0:
            raise ValueError("nearest_neighbors must be a positive integer")
        # dispatch to predict_single_sample, optionally in parallel
        if len(test_data) > 2000:
            with multiprocessing.Pool() as pool:
                params = [(test_sample, nearest_neighbors, calc_dist) for test_sample in test_data]
                preds = pool.starmap(self.predict_single_sample, params)
        else:
            preds = [self.predict_single_sample(row, nearest_neighbors, calc_dist) for row in test_data]
        return np.array(preds)

    def get_compression_ratio(self) -> float:
        """
        Estimate the compression ratio achieved by product quantisation.

        The compression ratio is defined as the ratio of the original vector dimensionality to the
        number of partition indices used to represent it.  Concretely, if the original vectors have
        ``d`` dimensions and are partitioned into ``n`` subvectors, then each compressed code
        consists of ``n`` integer indices.  The compression ratio is therefore ``d / n``.  A
        higher value indicates that more floating‑point values are represented by a single code.

        Returns
        -------
        float
            The estimated compression ratio.  If the model has not yet been trained (i.e.,
            ``self.d`` is not set), returns ``float('nan')``.
        """
        # ensure that the dimensionality is known
        if self.d is None or self.n is None or self.n == 0:
            return float('nan')
        try:
            # simple dimension-based ratio
            ratio = float(self.d) / float(self.n)
            return ratio
        except Exception:
            return float('nan')

    
    #Export of K‑Means training statistics
    def export_histories(self, dir_path: str) -> None:
        """
        Export per‑partition K‑Means statistics collected during compression.

        Each partition's metrics (inertia, n_iter, cluster_sizes, fit_time_sec) are
        written as a JSON file ``partition_{idx:02d}_kmeans_stats.json`` into
        ``dir_path``.  If no histories are available, the method does nothing.
 
        Parameters
        ----------
        dir_path : str
            Directory into which the JSON files are written.  The directory
            is created if it does not exist.
        """
        
        if not self.kmeans_histories:
            return
        os.makedirs(dir_path, exist_ok=True)
        for p_idx, metrics in self.kmeans_histories.items():
            try:
                path = os.path.join(dir_path, f"partition_{p_idx:02d}_kmeans_stats.json")
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2, ensure_ascii=False)
            except Exception:
                pass
