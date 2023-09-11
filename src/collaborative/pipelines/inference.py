import os
from typing import List, Tuple

from pyspark.ml import Pipeline
from pyspark.sql import SparkSession

from collaborative.nodes.inference_nodes import (
    CreateInferenceDataTransformer,
    ModelTransformer,
    RecommendationsTransformer,
)
from conf import catalog, globals, params, paths
from utils.experiment_tracking import MLflowHandler
from utils.psycopg_handler import PsycopgHandler


def run(
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

    ratings_path = paths.get_path(
        catalog.paths.DATA_01RAW,
        source,
        catalog.DatasetType.PRODUCTION,
        catalog.Datasets.RATINGS,
        suffix=catalog.FileFormat.PARQUET,
        storage=globals.Storage.S3,
        as_string=True,
    )

    ratings = session.read.parquet(ratings_path)

    # Convert the user_ids list to a DataFrame
    user_ids_df = session.createDataFrame(
        [(user_id,) for user_id in user_ids], ["userId"]
    )

    model = MLflowHandler.fetch_latest_model(model_name=model_name, stage=model_stage)

    # Init Inference Transformers
    create_inference_data_transformer = CreateInferenceDataTransformer(
        user_ids_df=user_ids_df
    )

    model_transformer = ModelTransformer(model=model)

    recommendations_transformer = RecommendationsTransformer(
        n_recommendations=n_recommendations
    )

    # Create pipeline
    pipeline = Pipeline(
        stages=[
            create_inference_data_transformer,
            model_transformer,
            recommendations_transformer,
        ]
    )

    # fit is necessary, but in this case it does nothing as we do not use Estimators
    pipeline = pipeline.fit(ratings)

    # Make predictions
    recommendations = pipeline.transform(ratings)
    recommendations: List[Tuple] = recommendations.collect()

    psycopg.update_db(
        f"""
            INSERT INTO {os.getenv("POSTGRES_RECOMMENDATIONS_TABLE")}
            (id, userid, movieid, prediction, rank, datetime)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (userid, rank) DO UPDATE
            SET id = EXCLUDED.id, movieid = EXCLUDED.movieid, prediction = EXCLUDED.prediction, rank = EXCLUDED.rank, datetime = EXCLUDED.datetime;
        """,
        recommendations,
    )
