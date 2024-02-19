from typing import Any, Dict, Optional, Tuple
from pathlib import Path

import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
# from mlflow.models import set_signature, infer_signature

from src.settings import Settings
from src.utils.pandas import pd_read, pd_write_random
from src.pipelines import Pipeline as src_Pipeline


class Train(src_Pipeline):
    def __init__(self, settings: Settings):
        self.settings: Settings = settings

    def run(self, training_file: Path, model_parameters: Dict = {}):
            """
            Runs the training pipeline.

            Args:
                raw_file (Optional[Path]): The path to the raw data file. If None, the default train raw file will be used.
                model_parameters (Dict): Optional dictionary of model parameters.
            """
            mlflow.sklearn.autolog(
                log_input_examples=True,
                log_datasets=False,
                log_model_signatures=False,
                registered_model_name=self.settings.MODEL_NAME,
            )
            df_train, df_eval = self._read_and_split(training_file)
            transformer = self._create_transformer()
            _, runid = self._train_mlflow(training_file, df_train, transformer, model_parameters)
            self._evaluate_mlflow(df_eval, runid)
            mlflow.sklearn.autolog(disable=True)

    def _create_transformer(self) -> ColumnTransformer:
        preprocessor = ColumnTransformer(
            [], remainder="drop", verbose_feature_names_out=False
        )

        numeric_transformer = Pipeline(steps=[("standardscaler", StandardScaler())])
        preprocessor.transformers.append(
            ("numeric", numeric_transformer, self.settings.NUMERIC_COLS)
        )

        nominal_transformer = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(missing_values=np.nan, strategy="most_frequent"),
                ),
                ("onehot", OneHotEncoder(drop="first", dtype="int")),
            ]
        )
        preprocessor.transformers.append(
            ("nominal", nominal_transformer, self.settings.CAT_NOM_COLS)
        )

        return preprocessor

    def _read_and_split(self, raw_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df_raw = pd_read(raw_file)

        X_train, X_test, y_train, y_test = train_test_split(
            df_raw.drop(columns=[self.settings.TARGET_COL]),
            df_raw[self.settings.TARGET_COL],
            test_size=0.2,
            random_state=42,
        )

        return (
            pd.concat([X_train, y_train], axis=1),
            pd.concat([X_test, y_test], axis=1),
        )

    def _train_mlflow(
        self,
        raw_file: Path,
        train_data: pd.DataFrame,
        transformer: Optional[Any],
        model_parameters: Dict = {},
    ) -> Tuple[Pipeline, str]:
        """
        Trains a model using MLflow and logs the training data and reference data.

        Args:
            raw_file (str): The path to the raw data file.
            train_data (pd.DataFrame): The training data.
            transformer (Optional[Any]): The transformer object for data preprocessing.
            model_parameters (Dict, optional): Additional parameters for the model. Defaults to {}.

        Returns:
            Tuple[Pipeline, str]: A tuple containing the trained model and the MLflow run ID.
        """
        with mlflow.start_run():
            model, _, y_predict = self._train_model(train_data, transformer, model_parameters)
            reference_file = pd_write_random(
                train_data.assign(prediction=y_predict),
                self.settings.REFERENCE_PATH,
                extension="csv",
            )
            training_dataset = mlflow.data.from_pandas(train_data,
                                              source=raw_file,
                                              targets=self.settings.TARGET_COL,
                                              name="training")
            logger.info(f"training file: {training_dataset.source.uri}")
            mlflow.log_input(dataset=training_dataset, context="training")
            reference_dataset = mlflow.data.from_pandas(train_data.assign(prediction=y_predict),
                                              source=reference_file,
                                              targets=self.settings.TARGET_COL,
                                              name="reference")
            logger.info(f"reference file: {reference_dataset.source.uri}")
            mlflow.log_input(dataset=reference_dataset, context="training")

            run_id = mlflow.active_run().info.run_id
            logger.info(f"MLflow run ID: {run_id}")
            return model, run_id

    def _evaluate_mlflow(self, df_eval: pd.DataFrame, runid: str):
            """
            Evaluate the MLflow model using the provided evaluation dataset.

            Args:
                df_eval (pd.DataFrame): The evaluation dataset.
                runid (str): The ID of the MLflow run.

            Returns:
                None
            """
            with mlflow.start_run(run_id=runid):
                model_uri = f"runs:/{runid}/model"
                mlflow.evaluate(model_uri, data=df_eval, targets=self.settings.TARGET_COL,
                    model_type="regressor", evaluator_config={"log_model_explainability": False})

    def _train_model(self, df_train: pd.DataFrame, pipeline: Pipeline, model_parameters: Dict = {}) -> Tuple[Pipeline, pd.DataFrame, np.ndarray]:
        """
        Trains a model using the provided training data and pipeline.

        Args:
            df_train (pd.DataFrame): The training dataset.
            pipeline (Pipeline): The data preprocessing pipeline.
            model_parameters (Dict, optional): Additional parameters for the model. Defaults to {}.

        Returns:
            Tuple[Pipeline, pd.DataFrame, np.ndarray]: A tuple containing the trained pipeline model, the input data, and the predicted output.
        """
        # Split the data into input(X) and output(y)
        y_train = df_train[self.settings.TARGET_COL]
        X_train = df_train.drop(columns=self.settings.TARGET_COL)

        # Train a Linear Regression Model with the train set
        model = RandomForestRegressor(random_state=0, **model_parameters)
        pipeline_model = make_pipeline(pipeline, model)
        pipeline_model.fit(X_train, y_train)

        # Train metrics are automatically logged when using mlflow.autolog
        y_predict = pipeline_model.predict(X_train)

        pipeline.set_output(transform="pandas")
        return pipeline_model, X_train, y_predict
