from pathlib import Path
import yaml

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


def load_yaml(yaml_path: Path) -> dict:
    """_summary_

    Parameters
    ----------
    yaml_path : Path
        _description_

    Returns
    -------
    dict
        _description_
    """

    assert isinstance(
        yaml_path, Path
    ), f"YAML path {yaml_path} must be an instance of pathlib.Path"

    with open(yaml_path, "r") as yaml_file:
        yaml_dict = yaml.safe_load(yaml_file)

    return yaml_dict
