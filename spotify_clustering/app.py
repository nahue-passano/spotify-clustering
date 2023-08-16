from pathlib import Path

import streamlit as st
import numpy as np

from spotify_clustering import utils
from spotify_clustering import eda

CSV_PATH = Path("spotify_clustering/data/spotify_dataset.csv")

st.set_page_config(layout="wide")
st.title("Spotify Dataset Analysis")


intro_tab, eda_tab, clustering_tab = st.tabs(
    ["Introduction", "Exploratory Data Analysis", "Clustering"]
)

with intro_tab:
    st.header("Introduction")

with eda_tab:
    st.header("Exploratory Data Analysis")

    spotify_df = utils.load_csv(CSV_PATH)

    st.subheader("Preview of the dataset")
    features = list(spotify_df.columns)
    spotify_numerical = spotify_df.select_dtypes(include=np.number)
    numerical_features = spotify_numerical.columns.tolist()
    st.write(
        f"The dataset has {len(features)} features, where {len(numerical_features)} are numerical features"
    )
    st.dataframe(spotify_df)

    st.subheader("Statistical metrics")
    spotify_stats = eda.dataframe_stats(spotify_df, ["kurt", "skew"])
    st.dataframe(spotify_stats)

    eda_fig = eda.make_histogram_from_dataframe(spotify_numerical)

    st.plotly_chart(eda_fig, height=800, use_container_width=True)

with clustering_tab:
    st.header("Clustering")
