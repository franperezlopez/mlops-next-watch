import mlflow
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import DataFrame

from conf import globals, params


def split_train_test(dataset: DataFrame):
    train, test = dataset.randomSplit([params.TRAIN, 1 - params.TRAIN], seed=42)
    return train, test


def train_als(train, val):
    def train_wrapper(params):
        with mlflow.start_run(nested=True):
            mlflow.set_tags(
                {
                    "model": globals.MLflow.ALS_MODEL_NAME,
                    "mlflow.runName": f"als_rank_{params['rank']}_reg_{params['reg_param']:4f}",
                }
            )
            mlflow.log_params(params)

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

            mlflow.log_metric("rmse", rmse)
            mlflow.spark.log_model(
                model,
                globals.MLflow.ALS_MODEL_ARTIFACT_PATH,
                registered_model_name=globals.MLflow.ALS_REGISTERED_MODEL_NAME,
            )

            return {"loss": rmse, "params": params, "status": STATUS_OK}

    return train_wrapper


def hyperparam_opt_als(train, val):
    trials = Trials()  # TODO: use `SparkTrials`` distribute trials to the workers

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

    with mlflow.start_run(run_name=globals.MLflow.ALS_RUN_NAME) as run:
        best = fmin(
            fn=train_als(train=train, val=val),
            space=space_search,
            algo=tpe.suggest,
            max_evals=5,
            trials=trials,
        )

        best_trial = sorted(trials.results, key=lambda result: result["loss"])[0]
        mlflow.log_dict(best_trial, "best_params.json")  # TODO: Change to log_params...
        return best_trial
