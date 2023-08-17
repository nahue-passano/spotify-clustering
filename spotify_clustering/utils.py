from pathlib import Path

import pandas as pd


def load_csv(csv_path: Path) -> pd.DataFrame:
    """Reads the csv data and return it as a dataframe

    Parameters
    ----------
    csv_path : Path
        Path to the csv

    Returns
    -------
    pd.DataFrame
        Csv data in dataframe object
    """
    return pd.read_csv(csv_path, sep=",")
