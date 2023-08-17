import re
from typing import List

import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots


def dataframe_stats(dataframe: pd.DataFrame, stats: List[str]) -> pd.DataFrame:
    """Generates statistical metrics for a given dataframe. Also more stats metrics can
    be added with "stats" variable

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame to be analyzed
    stats : List[str]
        List of strings where each element must be allowed by pd.DatFrame.agg() method

    Returns
    -------
    pd.DataFrame
        Statistical description of the dataframe
    """
    df_stats = dataframe.describe(include=[np.number])
    return pd.concat([df_stats, dataframe.select_dtypes(include=np.number).agg(stats)])


def make_histogram_from_dataframe(dataframe: pd.DataFrame) -> go.Figure:
    """Generates a plotly figure with the histogram of each numerical feature.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Numerical dataframe containing the data

    Returns
    -------
    go.Figure
        Hisogram of each dataframe feature.
    """
    # Init plotly figure
    fig = make_subplots(
        rows=4,
        cols=4,
        subplot_titles=dataframe.columns,
        vertical_spacing=0.1,
        horizontal_spacing=0.1,
    )

    for i, column_i in enumerate(dataframe.columns):
        # Position of each trace in figure
        row = i // 4 + 1
        col = i % 4 + 1

        trace = go.Histogram(x=dataframe[column_i], name=f"<b>{column_i}</b>")
        fig.add_trace(trace, row=row, col=col)

        # Update axes labels
        fig.update_xaxes(title_text="Value", row=row, col=col)
        fig.update_yaxes(title_text="Frequency", row=row, col=col)

    # Layout settings
    fig.update_layout(
        showlegend=False, margin={"l": 0, "r": 0, "t": 30, "b": 0}, height=800
    )

    return fig


def get_genres_from_dataframe(spotify_df: pd.DataFrame) -> List[str]:
    """Returns the unique genres of a given spotify dataframe

    Parameters
    ----------
    spotify_df : pd.DataFrame
        Spotify data in a Dataframe

    Returns
    -------
    List[str]
        List of strings containing unique values of genres
    """
    genres = [genre_i for genre_i in spotify_df["artist_genres"]]
    all_genres = _flatten_genre_list(genres)
    return _get_unique_genres(all_genres)


def _flatten_genre_list(genre_list: List[List[str]]) -> List[str]:
    """Flattens a given list of string list contaning artist genres into a single list.

    Parameters
    ----------
    genre_list : List[str]
        Genre list of list of string

    Returns
    -------
    List[str]
        Single list with all artist genres
    """
    flattened_genres = []

    for genre_str in genre_list:
        genres = re.findall(r"'([^']*)'", genre_str)
        genres = [genre.strip() for genre in genres if genre.strip() and genre != "[]"]
        flattened_genres.extend(genres)

    return flattened_genres


def _get_unique_genres(genre_list: List[str]) -> List[str]:
    """Returns a list of unique genres in the given genre_list

    Parameters
    ----------
    genre_list : List[str]
        List of genres

    Returns
    -------
    List[str]
        Unique genres in the given list.
    """
    return list(set(genre_list))
