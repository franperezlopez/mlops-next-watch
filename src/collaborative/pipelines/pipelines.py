from pyspark.sql import SparkSession

from collaborative.nodes import data_science_nodes, pre_processing_nodes
from conf import catalog, globals, params


def data_engineering(
    source: str = catalog.Sources.MOVIELENS,
    from_format: str = catalog.FileFormat.CSV,
    to_format: str = catalog.FileFormat.PARQUET,
    weights: list[float] = [1 - params.RAW, params.RAW],
    seed: int = params.SEED,
):
    session = (
        SparkSession.builder.appName("collab-de")
        .master("spark://localhost:7077")
        .config(map=params.Spark.spark_config)
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("INFO")
    pre_processing_nodes.make_raw_datasets(
        session,
        source,
        [
            catalog.Datasets.RATINGS,
            catalog.Datasets.TAGS,
        ],
        from_format,
        to_format,
        weights,
        seed,
        split_based_on_column="userId",
    )

    pre_processing_nodes.drop_columns(
        session,
        source,
        catalog.Datasets.RATINGS,
        catalog.DatasetType.TRAIN,
        *globals.DROP_COLUMNS,
    )

    pre_processing_nodes.drop_columns(
        session,
        source,
        catalog.Datasets.RATINGS,
        catalog.DatasetType.PRODUCTION,
        *globals.DROP_COLUMNS,
    )


def data_science(source: str = catalog.Sources.MOVIELENS):
    session = (
        SparkSession.builder.appName("collab-ds")
        .master("local[3]")
        .config(map=params.Spark.spark_config)
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("INFO")
    data_science_nodes.split_train_test(session, source)
    data_science_nodes.hyperparam_opt_als(session, source)
