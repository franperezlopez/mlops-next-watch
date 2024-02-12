import pytest
import pandas as pd

@pytest.fixture
def csv_file(tmp_path):
    file_path = tmp_path / "data.csv"
    data = {"col1": [1, 2, 3], "col2": [4, 5, 6]}
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    return file_path

@pytest.fixture
def parquet_file(tmp_path):
    file_path = tmp_path / "data.parquet"
    data = {"col1": [1, 2, 3], "col2": [4, 5, 6]}
    df = pd.DataFrame(data)
    df.to_parquet(file_path, index=False)
    return file_path
