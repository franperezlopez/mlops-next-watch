from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import DataFrame

from conf import params


def split_train_test(dataset: DataFrame):
    train, test = dataset.randomSplit([params.TRAIN, 1 - params.TRAIN], seed=42)
    return train, test


def train_als(train, val):
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
            metricName="rmse", labelCol="rating", predictionCol="prediction"
        )

        als_model = Pipeline(stages=[als])

        model = als_model.fit(train)

        predictions = model.transform(val)

        rmse = evaluator.evaluate(predictions)

        return {"loss": rmse, "params": params, "status": STATUS_OK}

    return train_wrapper


def hyperparam_opt_als(train, val):
    trials = Trials()  # TODO: use `SparkTrials`` distribute trials to the workers
    params = {
        "rank": hp.choice("rank", [8, 10, 12, 15, 18, 20]),
        "reg_param": hp.uniform("reg_param", 0.001, 0.2),
        "cold_start_strategy": hp.choice(
            "cold_start_strategy", ["drop"]
        ),  # ["nan", "drop"]
        "max_iter": hp.choice("max_iter", [5]),
    }
    best = fmin(
        fn=train_als(train=train, val=val),
        space=params,
        algo=tpe.suggest,
        max_evals=5,
        trials=trials,
    )
    best_trial = sorted(trials.results, key=lambda result: result["loss"])[0]
    return best_trial
