import pandas as pd
from pathlib import Path
import pytest
from src.utils.pandas import pd_read, pd_write, pd_write_random

def test_pd_read_csv(csv_file):
    df = pd_read(csv_file)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert list(df.columns) == ["col1", "col2"]

def test_pd_read_parquet(parquet_file):
    df = pd_read(parquet_file)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert list(df.columns) == ["col1", "col2"]

def test_pd_read_incorrect_format():
    with pytest.raises(ValueError):
        pd_read(Path("data.txt"))

def test_pd_write_csv(tmp_path):
    # Create a temporary Parquet file
    csv_file = tmp_path / "data.csv"

    # Create a sample DataFrame
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})

    # Write the DataFrame to the CSV file
    pd_write(df, csv_file)

    # Read the CSV file
    df_read = pd_read(csv_file)

    # Check if the DataFrame is the same as the original
    assert df.equals(df_read)

def test_pd_write_parquet(tmp_path):
    # Create a temporary Parquet file
    parquet_file = tmp_path / "data.parquet"

    # Create a sample DataFrame
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})

    # Write the DataFrame to the Parquet file
    pd_write(df, parquet_file)

    # Read the Parquet file
    df_read = pd_read(parquet_file)

    # Check if the DataFrame is the same as the original
    assert df.equals(df_read)

def test_pd_write_incorrect_format(tmp_path):
    # Create a temporary text file
    text_file = tmp_path / "data.txt"

    # Create a sample DataFrame
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})

    # Check if ValueError is raised when writing to an incorrect format
    with pytest.raises(ValueError):
        pd_write(df, text_file)

def test_pd_write_random_csv(tmp_path):
    # Create a temporary directory
    directory = tmp_path

    # Create a sample DataFrame
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})

    # Write the DataFrame to a random CSV file in the directory
    filepath = pd_write_random(df, directory)

    # Read the CSV file
    df_read = pd_read(filepath)

    # Check if the DataFrame is the same as the original
    assert df.equals(df_read)

def test_pd_write_random_parquet(tmp_path):
    # Create a temporary directory
    directory = tmp_path

    # Create a sample DataFrame
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})

    # Write the DataFrame to a random Parquet file in the directory
    filepath = pd_write_random(df, directory, extension="parquet")

    # Read the Parquet file
    df_read = pd_read(filepath)

    # Check if the DataFrame is the same as the original
    assert df.equals(df_read)