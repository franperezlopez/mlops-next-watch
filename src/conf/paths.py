from pathlib import Path
from typing import Union

from conf import globals

"""
Paths for the Next Watch project
"""

# BASE =   # if docker_running else   #
_SRC = Path("src")
_DATA = Path("data")
DATA_01EXTERNAL = _DATA / "01-external"
DATA_01RAW = _DATA / "01-raw"
DATA_O2INTERMEDIATE = _DATA / "02-intermediate"
DATA_03PROCESSED = _DATA / "03-processed"
DATA_04TRAIN = _DATA / "04-train"
LOGS = "logs"
ENV = ".env"
ASSETS = "assets"


def get_path(
    *args,
    suffix: str = "",
    as_string: bool = False,
    storage: str = globals.Storage.DOCKER
) -> Union[Path, str]:  # |
    """
    params:
        *args: can be of a Paths obj, a Source obj, a name of a dir as a string, a filename, etc.
        scope: the file's scope. For instance if a file is within project directories then the scope is the PROJECT, otherwise insert the root dir
    """
    path = Path()
    if storage == globals.Storage.DOCKER:
        path = Path("/app")
    if storage == globals.Storage.S3:
        path = Path()
    else:
        path = Path(__file__).parent.parent.parent.absolute()

    for arg in args:
        path = path / arg
    if suffix != "":
        path = path.with_suffix(suffix)

    # print(path.as_uri())
    return (
        ("s3a://" + str(path) if storage == globals.Storage.S3 else str(path))
        if as_string
        else path
    )
    # return (path.as_uri() if isinstance(path, S3Path) else str(path)) if as_string else path
