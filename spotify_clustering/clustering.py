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
        self.algorithm.fit(data)
        labels_df = pd.DataFrame({"cluster_labels": self.algorithm.labels_})
        return pd.concat([data, labels_df], axis=1)


class KMeansAlgorithm(ClusteringAlgorithm):
    def __init__(self, n_clusters: int):
        self.n_clusters = n_clusters
        self.max_iter = 500
        self.n_init = 50
        self.random_state = 42

    def set_algorithm(self):
        algorithm = KMeans(
            n_clusters=self.n_clusters,
            max_iter=self.max_iter,
            n_init=self.n_init,
            random_state=self.random_state,
        )
        super().__init__(algorithm)


class DBSCANAlgorithm(ClusteringAlgorithm):
    def __init__(self):
        self.eps = 0.3
        self.min_samples = 5

    def set_algorithm(self):
        algorithm = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        super().__init__(algorithm)


if __name__ == "__main__":
    from spotify_clustering import utils
    from pathlib import Path
    from spotify_clustering.dimensionality_reduction import PCATechnique, UMAPTechnique
    from spotify_clustering import visualization
    import numpy as np

    CSV_PATH = Path("spotify_clustering/data/spotify_dataset.csv")
    TECHNIQUE = "UMAP"
    CLUSTERING_ALGORITHM = "DBSCAN"

    spotify_df = utils.load_csv(CSV_PATH)
    spotify_numerical = spotify_df.select_dtypes(include=np.number)

    if TECHNIQUE == "PCA":
        dim_reductor = PCATechnique()
        axes = "PCA"
    if TECHNIQUE == "UMAP":
        dim_reductor = UMAPTechnique()
        axes = "UMAP"

    preprocessed_spotify = dim_reductor.preprocess(spotify_numerical)
    spotify_dim_reducted = dim_reductor.reduce_dimensionality(preprocessed_spotify, 3)

    if CLUSTERING_ALGORITHM == "KMEANS":
        cluster_algorithm = KMeansAlgorithm(n_clusters=5)
    if CLUSTERING_ALGORITHM == "DBSCAN":
        cluster_algorithm = DBSCANAlgorithm()
    df = cluster_algorithm.compute(spotify_dim_reducted)
    fig = visualization.make_scatter3d_from_dataframe(
        df,
        x_axes=f"{axes}1",
        y_axes=f"{axes}2",
        z_axes=f"{axes}3",
        hoverdata=spotify_df[["artist", "artist_genres"]],
        color_by="cluster_labels",
    )
    fig.show()
