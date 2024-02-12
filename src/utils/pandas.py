from pathlib import Path
from typing import Any

import pandas as pd
import uuid

def pd_read(input_file: Path) -> pd.DataFrame:
    """
    Read a file into a pandas DataFrame.

    Args:
        input_file (Path): The path to the input file.

    Returns:
        pd.DataFrame: The DataFrame containing the data from the input file.

    Raises:
        ValueError: If the input file has an incorrect format.
    """
    match input_file.suffix:
        case ".csv":
            df = pd.read_csv(input_file)
        case ".parquet":
            df = pd.read_parquet(input_file)
        case _:
            raise ValueError("input_file incorrect format")

    return df


def pd_write(df: pd.DataFrame, output_file: Path, **pandas_kwargs) -> pd.DataFrame:
    """
    Write a pandas DataFrame to a file.

    Args:
        df (pd.DataFrame): The DataFrame to be written.
        output_file (Path): The path to the output file.
        **pandas_kwargs: Additional keyword arguments to be passed to the pandas writer.

    Returns:
        pd.DataFrame: The output DataFrame.

    Raises:
        ValueError: If the output_file has an incorrect format.
    """
    match output_file.suffix:
        case ".csv":
            df.to_csv(output_file, index=False, **pandas_kwargs)
        case ".parquet":
            df.to_parquet(output_file, **pandas_kwargs)
        case _:
            raise ValueError("output_file incorrect format")

    return output_file


def pd_write_random(df: pd.DataFrame, directory: Path, extension: str = "csv", **pandas_kwargs) -> Path:
    """
    Write a pandas DataFrame to a random file in the specified directory.

    Args:
        df (pd.DataFrame): The DataFrame to be written.
        directory (Path): The directory where the file will be saved.
        extension (str, optional): The file extension. Defaults to "csv".
        **pandas_kwargs: Additional keyword arguments to be passed to the pandas write function.

    Returns:
        str: The filepath of the saved file.
    """
    filepath = directory / f'{str(uuid.uuid4()).replace("-", "")}.{extension}'
    pd_write(df, filepath, **pandas_kwargs)
    return filepath
