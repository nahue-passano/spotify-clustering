from pathlib import Path

import pandas as pd


def load_csv(csv_path: Path) -> pd.DataFrame:
    """_summary_

    Parameters
    ----------
    csv_path : Path
        _description_

    Returns
    -------
    pd.DataFrame
        _description_
    """
    return pd.read_csv(csv_path, sep=",")
