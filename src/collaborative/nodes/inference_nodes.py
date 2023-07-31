from typing import List

import mlflow.pyfunc
import mlflow.spark
from pyspark.sql import DataFrame, Row, SparkSession
from pyspark.sql.functions import col

from conf import catalog, globals, paths


def create_inference_data(session: SparkSession, source: str, user_id: int):
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

    user_ratings: DataFrame = ratings.filter(col("userId") == user_id)

    # Step 2: Collect all the movieIds that were rated by the given userId
    rated_movie_ids: List = (
        user_ratings.select("movieId").rdd.flatMap(lambda x: x).collect()
    )

    # Step 3: Filter the "ratings" DataFrame to get movieIds that weren't rated by the given userId
    unrated_movies: DataFrame = ratings.filter(~col("movieId").isin(rated_movie_ids))

    # Collect the movieIds that weren't rated by the given userId into a list
    unrated_movie_ids: List = (
        unrated_movies.select("movieId").rdd.flatMap(lambda x: x).collect()
    )

    # Create inference RDD
    inference_rdd = session.sparkContext.parallelize(
        [(user_id, movie_id) for movie_id in unrated_movie_ids]
    ).map(lambda x: Row(userId=int(x[0]), movieId=int(x[1])))

    # Transform to Inference DataFrame
    inference_data: DataFrame = session.createDataFrame(inference_rdd).select(
        ["userId", "movieId"]
    )

    return inference_data


def fetch_latest_model(model_name: str, stage: str):
    """Fetch the latest model from MLFlow in a given stage

    Returns:
        - a model instance
    """
    print("\n\na\n\n")
    # model = mlflow.spark.load_model(model_uri=f"models:/{model_name}/{stage}", dfs_tmpdir="s3://mlflow/tmp")dst_path
    model = mlflow.spark.load_model(
        model_uri=f"models:/{model_name}/{stage}",
        dfs_tmpdir="/sparktmp",
        dst_path="/sparktmp",
    )
    print("\n\nb\n\n")
    return model


def recommend_movies(model, inference_data: DataFrame, n_recommendations: int):
    """Recommends movies to a user

    Returns:
        - `n` movie recommendations
    """
    return (
        model.transform(inference_data)
        .select(["movieId", "prediction"])
        .orderBy("prediction", ascending=False)
        .rdd.map(lambda r: (r[0], r[1]))
        .take(n_recommendations)
    )


# def create_inference_data_for_users(session: SparkSession, source: str, user_ids: List[int]):
#    ratings_path = paths.get_path(
#        catalog.paths.DATA_01RAW,
#        source,
#        catalog.DatasetType.PRODUCTION,
#        catalog.Datasets.RATINGS,
#        suffix=catalog.FileFormat.PARQUET,
#        storage=globals.Storage.S3,
#        as_string=True,
#    )
#    ratings = session.read.parquet(ratings_path)
#
#    # Convert the user_ids list to a DataFrame
#    user_ids_df = session.createDataFrame([(user_id,) for user_id in user_ids], ["userId"])
#
#    # Get all movieIds for each user
#    user_rated_movie_ids = ratings.join(user_ids_df, on="userId").select("userId", "movieId")
#
#    # Get all distinct movieIds in the ratings DataFrame
#    all_movie_ids = ratings.select("movieId").distinct()
#
#    # Get unrated movieIds for each user
#    unrated_movies = all_movie_ids.join(user_rated_movie_ids, on="movieId", how="left_anti")
#
#    # Cross join unrated movieIds with the user_ids to get inference data
#    inference_data = unrated_movies.crossJoin(user_ids_df)
#
#    return inference_data
