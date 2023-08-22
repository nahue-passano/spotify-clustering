from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
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

MAX_N_CLUSTERS = 13


def kmeans_metrics(
    data: pd.DataFrame,
    scaler=StandardScaler(),
    dim_reduction_technique="UMAP",
    max_n_clusters: int = MAX_N_CLUSTERS,
    plot_metrics: bool = True,
):
    """_summary_

    Parameters
    ----------
    data : pd.DataFrame
        _description_
    scaler : _type_, optional
        _description_, by default StandardScaler()
    dim_reduction_technique : str, optional
        _description_, by default "UMAP"
    max_n_clusters : int, optional
        _description_, by default MAX_N_CLUSTERS
    plot_metrics : bool, optional
        _description_, by default True

    Returns
    -------
    _type_
        _description_
    """
    if dim_reduction_technique == "PCA":
        dim_reductor = PCATechnique()

    if dim_reduction_technique == "UMAP":
        dim_reductor = UMAPTechnique(**dim_reduction_settings["umap"])

    data_preprocessed = dim_reductor.preprocess(data, scaler)
    data_dim_reduced = dim_reductor.reduce_dimensionality(data_preprocessed, 3)

    sum_squared_error = []
    silhouette_coefficient = []

    k_clusters = np.arange(2, max_n_clusters + 1)

    for k in k_clusters:
        clustering_settings["kmeans"]["n_clusters"] = k
        cluster_algorithm = KMeansAlgorithm(**clustering_settings["kmeans"])
        cluster_algorithm.compute(data_dim_reduced)

        sum_squared_error.append(cluster_algorithm.algorithm.inertia_)
        silhouette_coefficient.append(
            silhouette_score(data_dim_reduced, cluster_algorithm.algorithm.labels_)
        )

    if plot_metrics:
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(
                x=k_clusters,
                y=sum_squared_error,
                name="Sum of Squared Error (SSE)",
            ),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=k_clusters,
                y=silhouette_coefficient,
                name="Silhouette Coefficient",
            ),
            secondary_y=True,
        )
        fig.update_yaxes(
            title_text="<b>Sum of Squared Error (SSE)</b>", secondary_y=False
        )
        fig.update_yaxes(title_text="<b>Silhouette Coefficient</b>", secondary_y=True)
        fig.update_layout(xaxis_title="n_clusters")
        fig.show()

    return sum_squared_error, silhouette_coefficient


if __name__ == "__main__":
    import numpy as np

    spotify_numerical = spotify_df.select_dtypes(include=np.number)

    sse_kmeans, solhouette_kmeans = kmeans_metrics(spotify_numerical)

    dim_reductor = UMAPTechnique(**dim_reduction_settings["umap"])
    data_preprocessed = dim_reductor.preprocess(spotify_numerical, StandardScaler())
    data_dim_reduced = dim_reductor.reduce_dimensionality(data_preprocessed, 3)
    cluster_algorithm = DBSCANAlgorithm(**clustering_settings["dbscan"])
    cluster_algorithm.compute(data_dim_reduced)
    print(
        f"Silhouette score for DBSCAN: {silhouette_score(data_dim_reduced, cluster_algorithm.algorithm.labels_)}"
    )
