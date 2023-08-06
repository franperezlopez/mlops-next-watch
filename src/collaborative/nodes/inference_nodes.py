import os
import pickle
from typing import List

import kafka
import mlflow.pyfunc
import mlflow.spark
from pyspark.sql import DataFrame, Row, SparkSession
from pyspark.sql.functions import col, rank
from pyspark.sql.window import Window

from conf import catalog, globals, paths


def create_inference_data_for_users(
    session: SparkSession, source: str, user_ids: List[int]
):
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

    # Get all movieIds for each user
    user_rated_movie_ids = ratings.join(user_ids_df, on="userId").select(
        "userId", "movieId"
    )

    # Get all distinct movieIds in the ratings DataFrame
    all_movie_ids = ratings.select("movieId").distinct()

    # Get unrated movieIds for each user
    unrated_movies = all_movie_ids.join(
        user_rated_movie_ids, on="movieId", how="left_anti"
    )

    # Cross join unrated movieIds with the user_ids to get inference data
    inference_data = unrated_movies.crossJoin(user_ids_df)

    return inference_data


def fetch_latest_model(model_name: str, stage: str):
    """Fetch the latest model from MLFlow in a given stage

    Returns:
        - a model instance
    """
    model = mlflow.spark.load_model(
        model_uri=f"models:/{model_name}/{stage}",
        dfs_tmpdir="/sparktmp",
        dst_path="/sparktmp",
    )
    return model


def recommend_movies(
    model,
    kafka_prod: kafka.KafkaProducer,
    kafka_topic: str,
    inference_data: DataFrame,
    n_recommendations: int,
):
    """Recommends movies to a user

    Returns:
        - `n` movie recommendations
    """
    predictions = model.transform(inference_data)
    window = Window.partitionBy(predictions["userId"]).orderBy(
        predictions["prediction"].desc()
    )
    recommendations = (
        predictions.select(
            col("userId"),
            col("movieId"),
            col("prediction"),
            rank().over(window).alias("rank"),
        )
        .filter(col("rank") <= n_recommendations)
        .rdd.map(lambda r: (r[0], r[1], r[2], r[3]))
        .collect()
    )

    kafka_prod.send(
        kafka_topic, pickle.dumps(recommendations)
    )
    kafka_prod.flush()

    return recommendations
