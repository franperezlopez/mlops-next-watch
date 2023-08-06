import os

import kafka
from pyspark.sql import SparkSession

from collaborative.nodes import (
    data_science_nodes,
    inference_nodes,
    pre_processing_nodes,
)
from conf import catalog, globals, params
from utils.psycopg_handler import PsycopgHandler


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
    user_ids: list[int],
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

    producer = kafka.KafkaProducer(
        bootstrap_servers=f"{os.environ['KAFKA_IP']}:{os.environ['KAFKA_PORT']}",
        api_version=(3, 5, 1),
    )

    psycopg = PsycopgHandler(
        os.getenv("POSTGRES_USER"),
        os.getenv("POSTGRES_PASSWORD"),
        os.getenv("POSTGRES_IP"),
        os.getenv("POSTGRES_PORT"),
        os.getenv("POSTGRES_APP_DATABASE"),
    )
    if user_ids == []:
        user_ids = [
            utuple[0]
            for utuple in psycopg.read_db(
                f"SELECT * FROM {os.getenv('POSTGRES_USERS_TABLE')}"
            )
        ]

    inference_data = inference_nodes.create_inference_data_for_users(
        session, source, user_ids
    )
    model = inference_nodes.fetch_latest_model(model_name=model_name, stage=model_stage)
    inference_nodes.recommend_movies(
        model,
        producer,
        os.getenv("KAFKA_RECOMMENDATIONS_TOPIC"),
        inference_data,
        n_recommendations,
    )
