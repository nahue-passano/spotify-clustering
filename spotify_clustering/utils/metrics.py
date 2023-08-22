from pathlib import Path

import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator


from spotify_clustering.utils import utils
from spotify_clustering.engine.clustering import KMeansAlgorithm, DBSCANAlgorithm
from spotify_clustering.engine.dimensionality_reduction import (
    PCATechnique,
    UMAPTechnique,
)

CSV_PATH = Path("spotify_clustering/data/spotify_dataset.csv")
DIM_REDUCTION_YAML = Path("spotify_clustering/configs/dimensionality_reduction.yaml")
CLUSTERING_YAML = Path("spotify_clustering/configs/clustering.yaml")

spotify_df = utils.load_csv(CSV_PATH)
dim_reduction_settings = utils.load_yaml(DIM_REDUCTION_YAML)
clustering_settings = utils.load_yaml(CLUSTERING_YAML)

MAX_N_CLUSTERS = 10


def predict_kmeans_n_clusters(
    data: pd.DataFrame,
    scaler=StandardScaler(),
    dim_reduction_technique="UMAP",
    max_n_clusters: int = MAX_N_CLUSTERS,
    plot_sse_curve: bool = True,
):
    if dim_reduction_technique == "PCA":
        dim_reductor = PCATechnique()

    if dim_reduction_technique == "UMAP":
        dim_reductor = UMAPTechnique(**dim_reduction_settings["umap"])

    data_preprocessed = dim_reductor.preprocess(data, scaler)
    data_dim_reduced = dim_reductor.reduce_dimensionality(data_preprocessed, 3)

    sum_squared_error = []

    for k in range(1, max_n_clusters + 1):
        clustering_settings["kmeans"]["n_clusters"] = k
        cluster_algorithm = KMeansAlgorithm(**clustering_settings["kmeans"])
        cluster_algorithm.compute(data_dim_reduced)
        sum_squared_error.append(cluster_algorithm.algorithm.inertia_)

    predicted_clusters = KneeLocator(
        range(1, max_n_clusters + 1),
        sum_squared_error,
        curve="convex",
        direction="decreasing",
    ).elbow

    if plot_sse_curve:
        fig = px.line(y=sum_squared_error, x=np.arange(1, MAX_N_CLUSTERS + 1))
        fig.update_layout(
            xaxis_title="n_clusters", yaxis_title="Sum of Squared Error (SSE)"
        )
        fig.show()

    return predicted_clusters


if __name__ == "__main__":
    import numpy as np

    spotify_numerical = spotify_df.select_dtypes(include=np.number)
    print(f"Predicted K-Means clusters: {predict_kmeans_n_clusters(spotify_numerical)}")
