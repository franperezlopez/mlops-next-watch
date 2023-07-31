from pyspark.sql import SparkSession

from collaborative.nodes import (
    data_science_nodes,
    inference_nodes,
    pre_processing_nodes,
)
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
        split_based_on_column="",
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
        .config(map=params.Spark.spark_config)
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("INFO")
    data_science_nodes.split_train_test(session, source)
    data_science_nodes.hyperparam_opt_als(session, source)


def inference(
    user_id: int,
    n_recommendations: int,
    source: str = catalog.Sources.MOVIELENS,
    model_name: str = globals.MLflow.ALS_REGISTERED_MODEL_NAME,
    model_stage: str = "Production",
):
    session = (
        SparkSession.builder.appName("inference")
        .config(map=params.Spark.spark_config)
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("INFO")
    inference_data = inference_nodes.create_inference_data(session, source, user_id)
    model = inference_nodes.fetch_latest_model(model_name=model_name, stage=model_stage)
    recommendations = inference_nodes.recommend_movies(
        model, inference_data, n_recommendations
    )
    print("yo: ", recommendations)
    return recommendations
    # inference
