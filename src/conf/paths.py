import os
from pathlib import Path

"""
Paths for the Next Watch project
"""

BASE = Path(__file__).parent.parent.parent.absolute()  # 
SRC = BASE / "src"
DATA = BASE / "data"
DATA_01EXTERNAL = DATA / "01-external"
DATA_01RAW = DATA / "01-raw"
DATA_O2INTERMEDIATE = DATA / "02-intermediate"
DATA_03PROCESSED = DATA / "03-processed"
DATA_04TRAIN = DATA / "04-train"
LOGS = BASE / "logs"
ENV = BASE / ".env"


def get_path(*args, suffix: str = "", as_string: bool = False) -> Path | str:
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
