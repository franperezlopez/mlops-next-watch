from pathlib import Path


class Paths:
    """
    Paths for the Next Watch project
    """

    BASE = Path(__file__).parent.parent.parent.absolute()
    SRC = BASE / "src"
    DATA = BASE / "data"
    DATA_01EXTERNAL = DATA / "01-external"
    DATA_01RAW = DATA / "01-raw"
    DATA_O2INTERMEDIATE = DATA / "02-intermediate"
    DATA_03PROCESSED = DATA / "03-processed"


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
    SERVE = "serve"


def create_external_dataset(source: str):
    (Paths.DATA_01EXTERNAL / source).mkdir(parents=True, exist_ok=True)


def create_raw_dataset(source: str):
    (Paths.DATA_01RAW / source).mkdir(parents=True, exist_ok=True)


def get_dataset_path(*args, suffix: str = "", as_string: bool = False) -> Path | str:
    """
    params:
        *args: can be of a Paths obj, a Source obj, a name of a dir as a string, a filename, etc.
    """
    path = Path()
    for arg in args:
        path = path / arg
    if suffix != "":
        path = path.with_suffix(suffix)
    return str(path) if as_string else path
