import pickle
from typing import List

import kafka
import mlflow.pyfunc
import mlflow.spark
from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasInputCols, HasOutputCols
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import DataFrame, Row, SparkSession
from pyspark.sql.functions import col, rank
from pyspark.sql.window import Window


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


# def create_inference_data_for_users(ratings: DataFrame, user_ids_df: DataFrame):  #
#    # Get all movieIds for each user
#    user_rated_movie_ids = ratings.join(user_ids_df, on="userId").select(
#        "userId", "movieId"
#    )
#
#    # Get all distinct movieIds in the ratings DataFrame
#    all_movie_ids = ratings.select("movieId").distinct()
#
#    # Get unrated movieIds for each user
#    unrated_movies = all_movie_ids.join(
#        user_rated_movie_ids, on="movieId", how="left_anti"
#    )
#
#    # Cross join unrated movieIds with the user_ids to get inference data
#    inference_data = unrated_movies.crossJoin(user_ids_df)
#
#    print(inference_data.columns)
#
#    return inference_data


class CreateInferenceDataTransformer(
    Transformer, HasInputCols, HasOutputCols, DefaultParamsReadable, DefaultParamsWritable
):

    # Create user ids df param... (for input cols, we inherent from `HasInputCols`, for instance).
    user_ids_df = Param(
        Params._dummy(),
        "user_ids_df",
        "List of user ids in Spark DataFrame type to make the inference",
    )

    @keyword_only
    def __init__(
        self, inputCols: List[str] = ["userId", "movieId"], user_ids_df=None
    ) -> None:
        super(CreateInferenceDataTransformer, self).__init__()
        self._setDefault(inputCols=inputCols, user_ids_df=user_ids_df)
        kwargs = self._input_kwargs
        self.set_params(**kwargs)

    @keyword_only
    def set_params(self, inputCols=["userId", "movieId"], user_ids_df=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def set_user_ids(self, user_ids_df: DataFrame):
        return self._set(user_ids_df=user_ids_df)

    def get_user_ids_df(self):
        return self.getOrDefault(self.user_ids_df)

    def _transform(self, ratings: DataFrame):

        user_ids: DataFrame = self.get_user_ids_df()

        # Get all movieIds for each user
        user_rated_movie_ids = ratings.join(user_ids, on="userId").select(
            *self.getInputCols()
        )

        # Get all distinct movieIds in the ratings DataFrame
        all_movie_ids = ratings.select("movieId").distinct()

        # Get unrated movieIds for each user
        unrated_movies = all_movie_ids.join(
            user_rated_movie_ids, on="movieId", how="left_anti"
        )

        # Cross join unrated movieIds with the user_ids to get inference data
        inference_data = unrated_movies.crossJoin(user_ids)

        return inference_data

    # https://stackoverflow.com/questions/41399399/serialize-a-custom-transformer-using-python-to-be-used-within-a-pyspark-ml-pipel
    # https://csyhuang.github.io/2020/08/01/custom-transformer/
    # create Leap Frames: https://combust.github.io/mleap-docs/mleap-runtime/create-leap-frame.html


class ModelTransformer(
    Transformer, HasOutputCols, DefaultParamsReadable, DefaultParamsWritable
):

    model = Param(
        Params._dummy(),
        "model",
        "The model that will make the inference",
    )

    @keyword_only
    def __init__(self, model=None) -> None:
        super(ModelTransformer, self).__init__()
        self._setDefault(model=model)
        kwargs = self._input_kwargs
        self.set_params(**kwargs)

    @keyword_only
    def set_params(self, model=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def set_model(self, model):
        return self._set(model=model)

    def get_model(self):
        return self.getOrDefault(self.model)

    def _transform(self, inference_data: DataFrame):
        return self.get_model().transform(inference_data)


class RecommendationsTransformer(
    Transformer, HasOutputCols, DefaultParamsReadable, DefaultParamsWritable
):

    n_recommendations = Param(
        Params._dummy(),
        "n_recommendations",
        "Number of recommendations to get",
        TypeConverters.toInt,
    )

    @keyword_only
    def __init__(
        self, outputCols=["userId", "movieId", "prediction"], n_recommendations=5
    ) -> None:
        super(RecommendationsTransformer, self).__init__()
        self._setDefault(outputCols=outputCols, n_recommendations=n_recommendations)
        kwargs = self._input_kwargs
        self.set_params(**kwargs)

    @keyword_only
    def set_params(
        self, outputCols=["userId", "movieId", "prediction"], n_recommendations=5
    ):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def set_n_recommendations(self, n_recommendations):
        return self._set(n_recommendations=n_recommendations)

    def get_n_recommendations(self):
        return self.getOrDefault(self.n_recommendations)

    def _transform(self, predictions: DataFrame):
        window = Window.partitionBy(predictions["userId"]).orderBy(
            predictions["prediction"].desc()
        )
        recommendations = (
            predictions.select(
                *self.getOutputCols(),
                rank().over(window).alias("rank"),
            )
            .filter(col("rank") <= self.get_n_recommendations())
            .rdd.map(lambda r: (r[0], r[1], r[2], r[3]))
        )
        return recommendations


def recommend_movies(  # TODO: TRANSFORM THIS FUNCTION Into a Custom Transformer also...
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

    kafka_prod.send(kafka_topic, pickle.dumps(recommendations))
    kafka_prod.flush()

    return recommendations
