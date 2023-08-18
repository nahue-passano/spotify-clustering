from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
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
    def preprocess(
        dataframe: pd.DataFrame, scaler: sklearn.preprocessing = StandardScaler()
    ) -> pd.DataFrame:
        scaled_df = dataframe.copy()
        scaled_df[list(scaled_df.columns)] = scaler.fit_transform(dataframe)


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


if __name__ == "__main__":
    import plotly.express as px

    CSV_PATH = Path("spotify_clustering/data/spotify_dataset.csv")

    spotify_df = utils.load_csv(CSV_PATH)
    spotify_numerical = spotify_df.select_dtypes(include=np.number)

    dim_reductor = PCATechnique()
    preprocessed_spotify = dim_reductor.preprocess(spotify_numerical)
    spotify_dim_reducted = dim_reductor.reduce_dimensionality(preprocessed_spotify, 3)

    fig = px.scatter_3d(spotify_dim_reducted, x="PC1", y="PC2", z="PC3")
    fig.show()


# # preprocessing
# scaler = StandardScaler()
# spotify_scaled = spotify_numerical.copy()
# spotify_scaled[numerical_features] = scaler.fit_transform(spotify_numerical)
# # print(spotify_scaled.describe())
# target_dimension = 3
# principal_components_analyzer = PCA(n_components=target_dimension)
# spotify_pca = principal_components_analyzer.fit_transform(spotify_scaled)
# spotify_pca_df = pd.DataFrame(
#     data=spotify_pca, columns=[f"PCA{i+1}" for i in range(target_dimension)]
# )
# print(spotify_pca_df)
# print(pd.concat([spotify_scaled, spotify_pca_df], axis=1))

#

# fig = px.scatter_3d(spotify_pca_df, x="PC1", y="PC2", z="PC3")
# fig.show()
