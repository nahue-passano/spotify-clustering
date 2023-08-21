import re
from typing import List

import pandas as pd
import numpy as np


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
