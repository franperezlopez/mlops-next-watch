from pathlib import Path
from typing import Tuple
import pandas as pd
from src.services.mlflow import Mlflow
from src.settings import Settings
from src.utils.pandas import pd_read
from loguru import logger
import uuid
from src.pipelines import Pipeline


class Predict(Pipeline):
    def __init__(self, settings: Settings):
        self.settings: Settings = settings

    def run(self, predict_file: Path) -> str:
            """
            Runs the inference pipeline on the given file.

            Args:
                file_path (Path): The path to the input file.

            Returns:
                str: The path to the saved predictions file.
            """
            df = pd_read(predict_file)
            y_hat, runid = self._predict(df)
            self.settings.MONITOR_PATH.mkdir(parents=True, exist_ok=True)
            predict_file_path = self.settings.MONITOR_PATH / self._random_file_name()
            logger.info(f"Saving predictions to {predict_file_path}")
            df.assign(prediction=y_hat, runid=runid).to_csv(predict_file_path, index=False)
            return predict_file_path

    def _random_file_name(self, extension: str = "csv") -> str:
        return str(uuid.uuid4()).replace("-", "") + f".{extension}"

    def _predict(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """
        The function downloads a model from the model registry, uses it to make predictions on the input
        dataframe, and returns the dataframe with the predictions appended

        :param df: The dataframe to be predicted on
        :type df: pd.DataFrame
        :return: A dataframe with the predicted values
        """
        model, runid = Mlflow.download_model(self.settings.MODEL_NAME)

        y_hat = model.predict(df)
        return y_hat, runid
