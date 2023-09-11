from typing import List

from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasInputCols, HasOutputCols
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    current_timestamp,
    monotonically_increasing_id,
    rank,
    row_number,
)
from pyspark.sql.window import Window


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
            predictions.withColumn(
                "datetime", current_timestamp()  # date_format(, "yyyy/MM/dd hh:mm:ss")
            )
            .withColumn(
                "index",
                row_number().over(Window.orderBy(monotonically_increasing_id())),
            )
            .select(
                "index",
                *self.getOutputCols(),
                rank().over(window).alias("rank"),
                "datetime",
            )
            .rdd.map(lambda r: (r[0], r[1], r[2], r[3], r[4], r[5]))
        )
        return recommendations
