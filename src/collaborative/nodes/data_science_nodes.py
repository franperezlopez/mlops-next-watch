import mlflow
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import DataFrame, SparkSession

from conf import catalog, globals, params, paths
from utils.experiment_tracking import MLflowHandler


def _train_als(train, val):
    """Where the train is taken..."""

    @MLflowHandler.log_train
    def train_wrapper(params):
        als = ALS(
            userCol="userId",
            itemCol="movieId",
            ratingCol="rating",
            rank=params["rank"],
            regParam=params["reg_param"],
            coldStartStrategy=params["cold_start_strategy"],
            maxIter=params["max_iter"],
        )

        evaluator = RegressionEvaluator(
            metricName=globals.MLflow.ALS_METRIC,
            labelCol="rating",
            predictionCol="prediction",
        )

        als_model = Pipeline(stages=[als])

        model = als_model.fit(train)

        predictions = model.transform(val)

        metric = evaluator.evaluate(predictions)

        return metric, model

    return train_wrapper


def split_train_test(session: SparkSession, source: str):
    """Splits the training data into train and test sets"""
    dataset = session.read.parquet(
        paths.get_path(
            paths.DATA_02PROCESSED,
            source,
            catalog.DatasetType.TRAIN,
            catalog.Datasets.RATINGS,
            suffix=catalog.FileFormat.PARQUET,
            storage=globals.Storage.S3,
            as_string=True,
        )
    )
    train, test = dataset.randomSplit([params.TRAIN, 1 - params.TRAIN], seed=42)
    train.write.mode("overwrite").parquet(
        paths.get_path(
            paths.DATA_03TRAIN,
            source,
            catalog.DatasetType.TRAIN,
            catalog.Datasets.RATINGS,
            suffix=catalog.FileFormat.PARQUET,
            storage=globals.Storage.S3,
            as_string=True,
        )
    )
    test.write.mode("overwrite").parquet(
        paths.get_path(
            paths.DATA_03TRAIN,
            source,
            catalog.DatasetType.TEST,
            catalog.Datasets.RATINGS,
            suffix=catalog.FileFormat.PARQUET,
            storage=globals.Storage.S3,
            as_string=True,
        )
    )


def hyperparam_opt_als(session: SparkSession, source: str):
    """Performs hyper-parameter optimization

    Args:
        train (DataFrame): A `Spark` `DataFrame` containing the training set

    """

    rank = params.ALS.RANK
    reg_param = params.ALS.REG_INTERVAL
    cold_start_strat = params.ALS.COLD_START_STRATEGY
    max_iter = params.ALS.MAX_ITER

    space_search = {
        "rank": hp.choice("rank", rank),
        "reg_param": hp.uniform("reg_param", reg_param[0], reg_param[1]),
        "cold_start_strategy": hp.choice("cold_start_strategy", cold_start_strat),
        "max_iter": hp.choice("max_iter", max_iter),
    }

    train = session.read.parquet(
        paths.get_path(
            catalog.paths.DATA_03TRAIN,
            source,
            catalog.DatasetType.TRAIN,
            catalog.Datasets.RATINGS,
            suffix=catalog.FileFormat.PARQUET,
            storage=globals.Storage.S3,
            as_string=True,
        )
    )
    val = session.read.parquet(
        paths.get_path(
            catalog.paths.DATA_03TRAIN,
            source,
            catalog.DatasetType.TEST,
            catalog.Datasets.RATINGS,
            suffix=catalog.FileFormat.PARQUET,
            storage=globals.Storage.S3,
            as_string=True,
        )
    )

    trials = (
        Trials()
    )  # FROM `hyperopt` docs: Do not use the SparkTrials class with MLlib. SparkTrials is designed to distribute trials for algorithms that are not themselves distributed.

    _run_hyperparam_opt(train, val, space_search, trials)


@MLflowHandler.log_hyperparam_opt
def _run_hyperparam_opt(train, val, space_search, trials):
    # Start Hyper-Parameter Optimization
    best = fmin(
        fn=_train_als(train=train, val=val),
        space=space_search,
        algo=tpe.suggest,
        max_evals=5,
        trials=trials,
    )

    return best
