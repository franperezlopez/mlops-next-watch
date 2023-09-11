from abc import ABC, abstractclassmethod, abstractmethod
from typing import Union

import mlflow
import mlflow.data
from hyperopt import STATUS_OK, Trials
from mlflow import MlflowClient

from conf import globals, paths


class ExperimentTracking(ABC):
    @abstractclassmethod
    def log_train(cls, func):
        """
        Define a way to log the training...
        """

        def wrapper(params):
            pass

    @abstractclassmethod
    def log_hyperparam_opt(cls, func):
        """
        Define a way to log the hyper-parameter optimization....
        """

        def wrapper(train, val, space_search, trials):
            pass


class MLflowHandler(ExperimentTracking):
    @classmethod
    def log_train(cls, training_func):
        """
        MLflow training log implementation to be used as a decorator
        """

        def log_train_wrapper(params):
            with mlflow.start_run(nested=True):
                mlflow.set_tags(
                    {
                        "model": globals.MLflow.ALS_MODEL_NAME,
                        "mlflow.runName": f"als_rank_{params['rank']}_reg_{params['reg_param']:4f}",
                    }
                )
                mlflow.log_params(params)

                metric, model, predictions = training_func(params)

                mlflow.log_metric(globals.MLflow.ALS_METRIC, metric)
                mlflow.log_dict(predictions, "predictions.json")
                sparkml_tmp_dir = paths.get_path(
                    paths.SPARKML_TMP_DIR, storage=globals.Storage.DOCKER, as_string=True
                )
                mlflow.spark.log_model(
                    model,
                    globals.MLflow.ALS_MODEL_ARTIFACT_PATH,
                    dfs_tmpdir=sparkml_tmp_dir,
                )

                return {
                    "loss": metric,
                    "params": params,
                    "status": STATUS_OK,
                }

        return log_train_wrapper

    @classmethod
    def log_hyperparam_opt(cls, hyperparam_func):
        """
        Mlflow hyperparam optimization implementation to be used as a decorator
        """

        def log_hyperparam_wrapper(node_class, train, val, space_search, trials):
            with mlflow.start_run(run_name=globals.MLflow.ALS_RUN_NAME) as run:
                mlflow.log_input(mlflow.data.from_spark(train), context="training_data")
                mlflow.log_input(mlflow.data.from_spark(val), context="validation_data")

                best = hyperparam_func(node_class, train, val, space_search, trials)

                # Log best trail/run
                client, best_run_id, best_run_metric, best_run_params = MLflowHandler._log_best_trial(
                    run, trials
                )
                metric_of_model_in_registry = MLflowHandler._get_metric_of_model_in_registry(globals.MLflow.ALS_REGISTERED_MODEL_NAME, "Production")

                if best_run_metric > metric_of_model_in_registry:
                    # Register best model in Mlflow Model Registry
                    MLflowHandler._register_best_model(client, best_run_id)

        return log_hyperparam_wrapper

    @classmethod
    def _log_best_trial(
        cls, run: mlflow.entities.Run, trials: Trials
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
        best_metric = best_run.data.metrics[globals.MLflow.ALS_METRIC]
        mlflow.set_tag("best_run", best_run_id)
        mlflow.log_metric("best_metric", best_metric)
        mlflow.log_dict(
            best_trial_params, "best_params.json"
        )  # TODO: Change to log_params...
        # mlflow.log_dict(best_predictions, "best_predictions.json")
        return client, best_run_id, best_metric, best_trial_params

    @classmethod
    def _get_info_of_model_in_registry(cls, model_name: str, stage: str) :
        model_uri = f"models:/{model_name}/{stage}"
        model_info = mlflow.models.get_model_info(model_uri)
        return model_info
        
    @classmethod
    def _get_metric_of_model_in_registry(cls, model_name: str, stage: str) -> int:
        model_in_registry_info = MLflowHandler._get_info_of_model_in_registry(model_name, stage)
        model_metric_history = mlflow.tracking.MlflowClient().get_metric_history(model_in_registry_info.run_id, "rmse")
        try:
            model_metric = model_metric_history[0].value
        except Exception as exception:
            raise Exception(f"I wasn't able to get model metric from the registry, exception {exception} occurred.")

        return model_metric
            

    @classmethod
    def _register_best_model(cls, client: mlflow.tracking.MlflowClient, run_id: str):
        model_version = mlflow.register_model(
            f"runs:/{run_id}/{globals.MLflow.ALS_MODEL_ARTIFACT_PATH}",
            globals.MLflow.ALS_REGISTERED_MODEL_NAME,
        )
        client.transition_model_version_stage(
            name=globals.MLflow.ALS_REGISTERED_MODEL_NAME,
            version=model_version.version,
            stage="Production",
            archive_existing_versions=True,
        )

    @classmethod
    def get_artifacts_from_model(
        cls, model_name: str, stage: str, artifact_path: str, as_dict=True
    ) -> Union[dict, str]:
        model_info = MLflowHandler._get_info_of_model_in_registry(model_name, stage)
        artifact_local_path = mlflow.artifacts.download_artifacts(
            run_id=model_info.run_id, artifact_path=artifact_path
        )
        return (
            mlflow.artifacts.load_dict(artifact_local_path)
            if as_dict
            else artifact_local_path
        )

    @classmethod
    def fetch_latest_model(cls, model_name: str, stage: str):
        """Fetch the latest model from MLFlow in a given stage

        Returns:
            - a model instance
        """
        sparkml_tmp_dir = paths.get_path(
            paths.SPARKML_TMP_DIR, storage=globals.Storage.DOCKER, as_string=True
        )
        model = mlflow.spark.load_model(
            model_uri=f"models:/{model_name}/{stage}",
            dfs_tmpdir=sparkml_tmp_dir,
            dst_path=sparkml_tmp_dir,
        )
        return model
