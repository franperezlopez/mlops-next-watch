from pathlib import Path
from typing import Any

import pandas as pd
import uuid

def pd_read(input_file: Path) -> pd.DataFrame:
    """
    Read a file into a pandas DataFrame depending on the input_file extension.

    :param input_file: The path to the input file
    :type input_file: Path
    :return: A dataframe
    """
    if input_file.suffix == ".csv":
        df = pd.read_csv(input_file)
    elif input_file.suffix == ".parquet":
        df = pd.read_parquet(input_file)
    else:
        raise ValueError("input_file incorrect format")

    return df


def pd_write(df: pd.DataFrame, output_file: Path, **pandas_kwargs) -> pd.DataFrame:
    """
    Write a pandas DataFrame to a file using the format specified in the output_file extension
    
    :param df: pd.DataFrame
    :type df: pd.DataFrame
    :param output_file: The file path to write the data to
    :type output_file: Path
    :return: The output file
    """
    if output_file.suffix == ".csv":
        df.to_csv(output_file, index=False, **pandas_kwargs)
    elif output_file.suffix == ".parquet":
        df.to_parquet(output_file, **pandas_kwargs)
    else:
        raise ValueError("output_file incorrect format")

    return output_file

def pd_write_random(df: pd.DataFrame, directory: Path, extension: str = "csv", **pandas_kwargs) -> str:
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
