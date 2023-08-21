from abc import ABC, abstractmethod

import pandas as pd
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


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
        return data_pca_df


class UMAPTechnique(DimensionalityReductionTechnique):
    def __init__(self, n_neighbors, min_dist, random_state):
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.random_state = random_state

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
        data_umap = umap.fit_transform(data)
        data_umap_df = pd.DataFrame(
            data=data_umap[:, :3],
            columns=[f"UMAP{i+1}" for i in range(target_dimension)],
        )
        return data_umap_df
