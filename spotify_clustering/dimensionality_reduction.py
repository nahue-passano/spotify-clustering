from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from spotify_clustering import utils


class DimensionalityReductionTechnique(ABC):
    """Base interface for dimensionality reduction techniques"""

    @abstractmethod
    def reduce_dimensionality(self, target_dimension: int) -> pd.DataFrame:
        """Abstract method to be overwritten by a concrete dimensionality reduction
        technique"""

    @staticmethod
    def preprocess(dataframe: pd.DataFrame, scaler=StandardScaler()) -> pd.DataFrame:
        scaled_df = dataframe.copy()
        scaled_df[list(scaled_df.columns)] = scaler.fit_transform(dataframe)
        return scaled_df


class PCATechnique(DimensionalityReductionTechnique):
    def reduce_dimensionality(
        self, data: pd.DataFrame, target_dimension: int
    ) -> pd.DataFrame:
        """_summary_

        Parameters
        ----------
        data : pd.DataFrame
            _description_
        target_dimension : int
            _description_

        Returns
        -------
        pd.DataFrame
            _description_
        """
        principal_components_analyzer = PCA(n_components=target_dimension)
        data_pca = principal_components_analyzer.fit_transform(data)
        data_pca_df = pd.DataFrame(
            data=data_pca, columns=[f"PCA{i+1}" for i in range(target_dimension)]
        )
        return pd.concat([data, data_pca_df], axis=1)


class UMAPTechnique(DimensionalityReductionTechnique):
    def __init__(self):
        self.n_neighbors = 5
        self.min_dist = 0.0
        self.random_state = 42

    def reduce_dimensionality(
        self, data: pd.DataFrame, target_dimension: int
    ) -> pd.DataFrame:
        """_summary_

        Parameters
        ----------
        data : pd.DataFrame
            _description_
        target_dimension : int
            _description_

        Returns
        -------
        pd.DataFrame
            _description_
        """
        umap = UMAP(
            n_components=target_dimension,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            random_state=self.random_state,
        )
        data_pca = umap.fit_transform(data)
        data_pca_df = pd.DataFrame(
            data=data_pca[:, :3],
            columns=[f"UMAP{i+1}" for i in range(target_dimension)],
        )
        return pd.concat([data, data_pca_df], axis=1)


if __name__ == "__main__":
    import plotly.express as px

    CSV_PATH = Path("spotify_clustering/data/spotify_dataset.csv")
    TECHNIQUE = "UMAP"

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

    from spotify_clustering import visualization

    fig = visualization.make_scatter3d_from_dataframe(
        spotify_dim_reducted,
        x_axes=f"{axes}1",
        y_axes=f"{axes}2",
        z_axes=f"{axes}3",
        hoverdata=spotify_df[["artist", "artist_genres"]],
    )

    fig.show()
