from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from spotify_clustering.engine import utils
from spotify_clustering.engine import eda
from spotify_clustering.engine import visualization
from spotify_clustering.engine.dimensionality_reduction import (
    PCATechnique,
    UMAPTechnique,
)
from spotify_clustering.engine.clustering import KMeansAlgorithm, DBSCANAlgorithm

# TODO: agregar ploteo de la matriz de correlación
# TODO: Agregar gráfico temporal de como cambian los parámetros en función del tiempo

CSV_PATH = Path("spotify_clustering/data/spotify_dataset.csv")
DIMENSIONALITY_REDUCTION_YAML = Path(
    "spotify_clustering/configs/dimensionality_reduction.yaml"
)
CLUSTERING_YAML = Path("spotify_clustering/configs/clustering.yaml")

spotify_df = utils.load_csv(CSV_PATH)
dimensionality_reduction_settings = utils.load_yaml(DIMENSIONALITY_REDUCTION_YAML)
clustering_settings = utils.load_yaml(CLUSTERING_YAML)


def main():
    st.set_page_config(layout="wide")
    st.title("Spotify Dataset Analysis")

    intro_tab, eda_tab, clustering_tab = st.tabs(
        ["Introduction", "🔎🗃️ Exploratory Data Analysis", "Clustering"]
    )

    with intro_tab:
        intro_tab_code()

    with eda_tab:
        eda_tab_code()

    with clustering_tab:
        clustering_tab_code()


def intro_tab_code():
    st.header("Introduction")

    st.write("Tech stack used for the challenge:")
    st.write("* Pandas for data manipulation")
    st.write("* Plotly for data visualization")
    st.write(
        "* Scikit-learng for dimensionality reduction algorithms and clustering techniques"
    )
    st.write("* Streamlit for the front-end app")


def eda_tab_code():
    st.header("🔎🗃️ Exploratory Data Analysis")
    st.divider()

    st.subheader("Preview of the dataset")

    # Features
    features = list(spotify_df.columns)
    spotify_numerical = spotify_df.select_dtypes(include=np.number)
    numerical_features = spotify_numerical.columns.tolist()

    # General information of the dataset
    artists = spotify_df["artist"].unique()
    genres = eda.get_genres_from_dataframe(spotify_df)
    no_genre_artists = spotify_df[spotify_df["artist_genres"] == "['[]']"][
        "artist"
    ].unique()
    most_followed_artists = spotify_df[spotify_df["artist_followers"] > 10_000_000][
        "artist"
    ]
    most_followed_artists_pct = round(
        100 * len(most_followed_artists) / len(artists), 2
    )
    most_used_key = spotify_df["key"].mode()[0]

    st.write(
        f"* The dataset has {len(features)} features, where {len(numerical_features)} are numerical features"
    )
    st.write(f"* Has {len(genres)} different musical genres")
    st.write(
        f"* There are {len(artists)} different artists in the dataset, where {len(no_genre_artists)} have no assigned musical gender"
    )
    st.write(
        f"* The {most_followed_artists_pct}% of the artists has more than 10 million followers"
    )
    st.write(f"* The most used key is {most_used_key}, which corresponds to A")

    st.dataframe(spotify_df)
    st.divider()

    st.subheader("Statistical metrics")
    spotify_stats = eda.dataframe_stats(spotify_df, ["kurt", "skew"])
    st.dataframe(spotify_stats)
    st.divider()

    st.subheader("Features histogram")
    histogram_fig = visualization.make_histogram_from_dataframe(spotify_numerical)

    st.plotly_chart(histogram_fig, height=800, use_container_width=True)


def clustering_tab_code():
    st.header("Clustering Analysis")
    st.divider()
    prepro_col, dim_red_col, cluster_col = st.columns(3)
    with prepro_col:
        st.subheader("Data Preprocessing")
        scaler_select = st.selectbox(
            "Scaler", options=["StandardScaler", "MinMaxScaler"]
        )

        if scaler_select == "StandardScaler":
            scaler = StandardScaler()

        if scaler_select == "MinMaxScaler":
            scaler = MinMaxScaler()

    with dim_red_col:
        st.subheader("Dimensionality Reduction technique")
        dim_red_select = st.selectbox("Technique", options=["PCA", "UMAP"])

        if dim_red_select == "PCA":
            dim_reductor = PCATechnique()
            axes = "PCA"

        if dim_red_select == "UMAP":
            umap_settings = st.columns(2)

            n_neighbors = int(
                umap_settings[0].slider(
                    "n_neighbors",
                    3,
                    200,
                    dimensionality_reduction_settings["umap"]["n_neighbors"],
                )
            )
            min_dist = float(
                umap_settings[1].slider(
                    "min_dist",
                    0.0,
                    1.0,
                    dimensionality_reduction_settings["umap"]["min_dist"],
                )
            )
            random_state = int(
                dimensionality_reduction_settings["umap"]["random_state"]
            )

            dim_reductor = UMAPTechnique(n_neighbors, min_dist, random_state)
            axes = "UMAP"

    with cluster_col:
        st.subheader("Clustering algorithm")
        cluster_select = st.selectbox("Algorithm", options=["K-Means", "DBSCAN"])

        if cluster_select == "K-Means":
            kmeans_settings = st.columns(3)

            n_clusters = int(
                kmeans_settings[0].text_input(
                    "n_clusters", value=clustering_settings["kmeans"]["n_clusters"]
                )
            )
            max_iter = int(
                kmeans_settings[1].text_input(
                    "max_iter", value=clustering_settings["kmeans"]["max_iter"]
                )
            )
            n_init = int(
                kmeans_settings[2].text_input(
                    "n_init", value=clustering_settings["kmeans"]["n_init"]
                )
            )
            random_state = clustering_settings["kmeans"]["random_state"]

            cluster_algorithm = KMeansAlgorithm(
                n_clusters, max_iter, n_init, random_state
            )

        if cluster_select == "DBSCAN":
            dbscan_settings = st.columns(2)

            eps = float(
                dbscan_settings[0].text_input(
                    "eps", value=clustering_settings["dbscan"]["eps"]
                )
            )
            min_samples = int(
                dbscan_settings[1].text_input(
                    "min_samples", value=clustering_settings["dbscan"]["min_samples"]
                )
            )

            cluster_algorithm = DBSCANAlgorithm(eps, min_samples)

    spotify_numerical = spotify_df.select_dtypes(include=np.number)

    preprocessed_spotify = dim_reductor.preprocess(spotify_numerical, scaler=scaler)
    spotify_dim_reducted = dim_reductor.reduce_dimensionality(preprocessed_spotify, 3)

    df = cluster_algorithm.compute(spotify_dim_reducted)
    scatter3d_fig = visualization.make_scatter3d_from_dataframe(
        pd.concat([preprocessed_spotify, df], axis=1),
        x_axes=f"{axes}1",
        y_axes=f"{axes}2",
        z_axes=f"{axes}3",
        hoverdata=pd.concat(
            [spotify_df[["artist", "artist_genres"]], df["cluster_labels"]], axis=1
        ),
        color_by="cluster_labels",
    )

    st.plotly_chart(scatter3d_fig, height=800, use_container_width=True)


if __name__ == "__main__":
    main()
