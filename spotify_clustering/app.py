from pathlib import Path

import streamlit as st
import numpy as np

from spotify_clustering import utils
from spotify_clustering import eda

# TODO: agregar ploteo de la matriz de correlación
# TODO: Agregar gráfico temporal de como cambian los parámetros en función del tiempo

CSV_PATH = Path("spotify_clustering/data/spotify_dataset.csv")

st.set_page_config(layout="wide")
st.title("Spotify Dataset Analysis")


intro_tab, eda_tab, clustering_tab = st.tabs(
    ["Introduction", "🔎🗃️ Exploratory Data Analysis", "Clustering"]
)

with intro_tab:
    st.header("Introduction")
    
    st.write("Tech stack used for the challenge:")
    st.write("* Pandas for data manipulation")
    st.write("* Plotly for data visualization")
    st.write("* Scikit-learng for dimensionality reduction algorithms and clustering techniques")
    st.write("* Streamlit for the front-end app")

with eda_tab:
    st.header("🔎🗃️ Exploratory Data Analysis")
    st.divider()
    
    st.subheader("Preview of the dataset")
    spotify_df = utils.load_csv(CSV_PATH)
    
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
    eda_fig = eda.make_histogram_from_dataframe(spotify_numerical)

    st.plotly_chart(eda_fig, height=800, use_container_width=True)

with clustering_tab:
    st.header("Clustering")
