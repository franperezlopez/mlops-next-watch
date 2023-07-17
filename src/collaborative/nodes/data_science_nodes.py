import mlflow
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import DataFrame, SparkSession

from conf import catalog, globals, params, paths


def _train_als(train, val):
    """Where the train is taken..."""

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
                metricName=globals.MLflow.ALS_METRIC,
                labelCol="rating",
                predictionCol="prediction",
            )

            als_model = Pipeline(stages=[als])

            model = als_model.fit(train)

            predictions = model.transform(val)

            metric = evaluator.evaluate(predictions)

            mlflow.log_metric(globals.MLflow.ALS_METRIC, metric)
            mlflow.spark.log_model(
                model,
                globals.MLflow.ALS_MODEL_ARTIFACT_PATH,
            )

            return {"loss": metric, "params": params, "status": STATUS_OK}

    return train_wrapper


def _log_best_trial(
    run: mlflow.entities.Run, trials: Trials
) -> tuple[mlflow.tracking.MlflowClient, str, dict]:
    """Log the best trial
    Args:
        run (mlflow.entities.Run): An MLflow Run Object
        trials (Trials): all trials from hyperopt
    """
    client = mlflow.tracking.MlflowClient()
    best_trial_params = sorted(trials.results, key=lambda result: result["loss"])[0]

    runs = client.search_runs(
        [run.info.experiment_id], f"tags.mlflow.parentRunId = '{run.info.run_id}'"
    )
    best_run = min(runs, key=lambda run: run.data.metrics[globals.MLflow.ALS_METRIC])
    best_run_id = best_run.info.run_id
    mlflow.set_tag("best_run", best_run_id)
    mlflow.log_metric("best_metric", best_run.data.metrics[globals.MLflow.ALS_METRIC])
    mlflow.log_dict(
        best_trial_params, "best_params.json"
    )  # TODO: Change to log_params...

    return client, best_run_id, best_trial_params


def _register_best_model(client: mlflow.tracking.MlflowClient, run_id: str):
    model_version = mlflow.register_model(
        f"runs:/{run_id}/{globals.MLflow.ALS_MODEL_ARTIFACT_PATH}",
        globals.MLflow.ALS_REGISTERED_MODEL_NAME,
    )
    client.transition_model_version_stage(
        name=globals.MLflow.ALS_REGISTERED_MODEL_NAME,
        version=model_version.version,
        stage="Staging",
        archive_existing_versions=True,
    )


def split_train_test(session: SparkSession, source: str):
    """Splits the training data into train and test sets"""
    dataset = session.read.parquet(
        paths.get_path(
            paths.DATA_03PROCESSED,
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
            paths.DATA_04TRAIN,
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
            paths.DATA_04TRAIN,
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

    train = session.read.parquet(
        paths.get_path(
            catalog.paths.DATA_04TRAIN,
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
            catalog.paths.DATA_04TRAIN,
            source,
            catalog.DatasetType.TEST,
            catalog.Datasets.RATINGS,
            suffix=catalog.FileFormat.PARQUET,
            storage=globals.Storage.S3,
            as_string=True,
        )
    )
    # Start Hyper-Parameter Optimization
    with mlflow.start_run(run_name=globals.MLflow.ALS_RUN_NAME) as run:
        best = fmin(
            fn=_train_als(train=train, val=val),
            space=space_search,
            algo=tpe.suggest,
            max_evals=5,
            trials=trials,
        )

        # Log best trail/run
        client, best_run_id, best_trial_params = _log_best_trial(run, trials)

        # Register best model in Mlflow Model Registry
        _register_best_model(client, best_run_id)
