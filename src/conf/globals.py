from conf import catalog

"""
File Formats
"""
FROM_EXT_FORMAT = catalog.FileFormat.CSV
TO_RAW_FORMAT = catalog.FileFormat.PARQUET

"""
columns to drop
"""
DROP_COLUMNS = ["timestamp"]


class MLflow:
    """
    Experiment Tracking Parameters
    """

    EXPERIMENT_NAME: str = "COLLABORATIVE"

    ALS_MODEL_NAME: str = "ALS"
    ALS_MODEL_ARTIFACT_PATH: str = "als_model"
    ALS_RUN_NAME: str = "ALS HyperParam Optimization"
    ALS_REGISTERED_MODEL_NAME: str = "spark_als_model"
    ALS_METRIC = "rmse"


class Pipelines:
    """
    Pipeline Names
    """

    DATA_SCIENCE = "ds"
    DATA_ENGINEERING = "de"
    DS = "ds"
    DE = "de"
