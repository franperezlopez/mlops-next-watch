from pyspark.sql import SparkSession

from collaborative.nodes.data_processing_nodes import DataProcessingNodes
from conf import catalog, globals, params


def run(
    source: str = catalog.Sources.MOVIELENS,
    from_format: str = catalog.FileFormat.CSV,
    to_format: str = catalog.FileFormat.PARQUET,
    weights: list[float] = [1 - params.RAW, params.RAW],
    seed: int = params.SEED,
):
    session = (
        SparkSession.builder.appName("collab-de")
        .config(map=params.Spark.spark_config)
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("INFO")

    pre_processing_nodes = DataProcessingNodes(
        session, source, storage=globals.Storage.S3
    )
    pre_processing_nodes.make_raw_datasets(
        [
            catalog.Datasets.RATINGS,
            catalog.Datasets.TAGS,
        ],
        from_format,
        to_format,
        weights,
        seed,
        split_based_on_column="",
    )

    pre_processing_nodes.drop_columns(
        catalog.Datasets.RATINGS,
        catalog.DatasetType.TRAIN,
        *globals.DROP_COLUMNS,
    )

    pre_processing_nodes.drop_columns(
        catalog.Datasets.RATINGS,
        catalog.DatasetType.PRODUCTION,
        *globals.DROP_COLUMNS,
    )
