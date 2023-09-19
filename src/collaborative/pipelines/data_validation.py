from pyspark.sql import SparkSession

from collaborative.nodes.data_validation_nodes import DataValidation
from conf import catalog, globals, params


def run(
    source: str = catalog.Sources.MOVIELENS,
    dataset: str = catalog.Datasets.RATINGS,
    file_suffix: str = catalog.FileFormat.CSV,
    storage=globals.Storage.DOCKER,
):
    session = (
        SparkSession.builder.appName("collab-dv")
        .config(map=params.Spark.spark_config)
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("INFO")

    data_validation = DataValidation(session, source, storage)
    data_validation.validate_ratings(dataset, file_suffix)
