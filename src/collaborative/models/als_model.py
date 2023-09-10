from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from conf import globals


def get_model(params):
    als = ALS(
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        rank=params["rank"],
        regParam=params["reg_param"],
        coldStartStrategy=params["cold_start_strategy"],
        maxIter=params["max_iter"],
    )

    als_model = Pipeline(stages=[als])

    return als_model


def get_model_evaluator():
    evaluator = RegressionEvaluator(
        metricName=globals.MLflow.ALS_METRIC,
        labelCol="rating",
        predictionCol="prediction",
    )
    return evaluator
