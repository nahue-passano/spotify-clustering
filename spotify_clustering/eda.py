from pathlib import Path

import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from spotify_clustering.utils import load_csv

CSV_PATH = Path("spotify_clustering/data/spotify_dataset.csv")

spotify_df = load_csv(CSV_PATH)

# Vista previa
print("Vista previa del dataset")
print(spotify_df.head())

# Features
features = list(spotify_df.columns)
print("-" * 50)
print(f"El dataset tiene {len(features)} features, los cuales son: {features}")

numerical_features = spotify_df.select_dtypes(include=np.number).columns.tolist()
print("-" * 50)
print(
    f"De las {len(features)} features, {len(numerical_features)} son del tipo numéricas ({numerical_features})"
)

# Data cleaning check
print("-" * 50)
print(f"Chequeo si hace falta imputar data: {spotify_df.isnull().values.any()}")


# Stats description
def dataframe_stats(dataframe, stats):
    df_stats = dataframe.describe(include=[np.number])
    return pd.concat([df_stats, dataframe.select_dtypes(include=np.number).agg(stats)])


spotify_stats = dataframe_stats(spotify_df, ["kurt", "skew"])
print("-" * 50)
print(spotify_stats)

print("-" * 50)


spotify_numerical = spotify_df.select_dtypes(include=np.number)


def make_histogram_from_dataframe(dataframe: pd.DataFrame) -> go.Figure:
    # Crear una figura de subplots
    fig = make_subplots(
        rows=4,
        cols=4,
        subplot_titles=dataframe.columns,
        vertical_spacing=0.1,
        horizontal_spacing=0.1,
    )

    # Iterar a través de las columnas numéricas y agregar histogramas a la figura
    for i, column_i in enumerate(dataframe.columns):
        row = i // 4 + 1
        col = i % 4 + 1
        trace = go.Histogram(
            x=dataframe[column_i],
            name=f"<b>{column_i}</b>",
        )
        fig.add_trace(trace, row=row, col=col)
        fig.update_xaxes(title_text="Value", row=row, col=col)
        fig.update_yaxes(title_text="Frequency", row=row, col=col)

    # Actualizar el diseño de la figura para agregar etiquetas y título
    fig.update_layout(
        showlegend=False, margin={"l": 0, "r": 0, "t": 30, "b": 0}, height=800
    )

    return fig


if __name__ == "__main__":
    fig = make_histogram_from_dataframe(spotify_numerical)
    fig.show()
