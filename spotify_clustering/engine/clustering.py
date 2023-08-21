from abc import ABC, abstractmethod

import pandas as pd
from sklearn.cluster import KMeans, DBSCAN


class ClusteringAlgorithm(ABC):
    def __init__(self, algorithm):
        self.algorithm = algorithm

    @abstractmethod
    def set_algorithm(self):
        """blabla"""

    def compute(self, data: pd.DataFrame):
        self.set_algorithm()
        self.algorithm.fit_predict(data)
        labels_df = pd.DataFrame({"cluster_labels": self.algorithm.labels_})
        return pd.concat([data, labels_df], axis=1)


class KMeansAlgorithm(ClusteringAlgorithm):
    def __init__(self, n_clusters: int, max_iter: int, n_init: int, random_state: int):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state

    def set_algorithm(self):
        algorithm = KMeans(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            n_init=self.n_init,
            random_state=self.random_state,
        )
        super().__init__(algorithm)


class DBSCANAlgorithm(ClusteringAlgorithm):
    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples

    def set_algorithm(self):
        algorithm = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        super().__init__(algorithm)
