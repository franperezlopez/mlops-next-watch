from pathlib import Path
from typing import Any, Tuple
import mlflow


class Mlflow:
    _CLIENT = mlflow.tracking.MlflowClient()
    @staticmethod
    def download_model(registered_model_name: str = "model") -> Tuple[Any, str]:
        """
        It downloads the model from the MLflow model registry and returns the model object

        :param registered_model_name: The name of the registered model in MLflow, defaults to model
        :type registered_model_name: str (optional)
        :param version: The version of the model to download. If you don't specify this parameter, the
        :return: The model object
        """

        runid = Mlflow.get_latest_run_id(registered_model_name)
        model = mlflow.sklearn.load_model(model_uri=f"runs:/{runid}/model")
        return model, runid

    @staticmethod
    def get_latest_run_id(model_name: str) -> str:
        """
        It returns the run ID of the latest version of the model with the given name

        :param model_name: The name of the model
        :type model_name: str
        :return: The run_id of the latest version of the model.
        """
        reg_models = Mlflow._CLIENT.search_registered_models(f"name = '{model_name}'")
        return list(map(lambda x: x.latest_versions[0].run_id, reg_models))[0]

    @staticmethod
    def get_dataset_path(runid: str) -> Path:
        """
        It returns the dataset file path for a given runid

        :param runid: The runid of the model
        :type runid: str
        :return: The file path of the dataset
        """
        run = Mlflow._CLIENT.get_run(runid)
        for input in run.inputs.dataset_inputs:
            if input.dataset.name == "reference":
                dataset_source = mlflow.data.get_source(input.dataset)
                return Path(dataset_source.uri)