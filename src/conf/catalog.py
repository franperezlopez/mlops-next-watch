from conf import paths

"""
Data Catalog
"""


class Sources:
    """
    Data Sources
    """

    MOVIELENS = "movielens"


class Datasets:
    """
    Datasets filenames
    """

    RATINGS = "ratings"
    TAGS = "tags"
    MOVIES = "movies"
    LINKS = "links"


class FileFormat:
    """
    File Format
    """

    CSV = ".csv"
    PARQUET = ".parquet"


class DatasetType:
    """
    Dataset Types
    """

    TRAIN = "train"
    TEST = "test"
    PRODUCTION = "prod"


def create_external_dataset(source: str):
    (paths.DATA_01EXTERNAL / source).mkdir(parents=True, exist_ok=True)


def create_raw_dataset(source: str):
    (paths.DATA_01RAW / source).mkdir(parents=True, exist_ok=True)
