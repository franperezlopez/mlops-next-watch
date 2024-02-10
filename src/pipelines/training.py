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

    def run(self, raw_file: Optional[Path], model_parameters: Dict = {}):
            """
            Runs the training pipeline.

            Args:
                raw_file (Optional[Path]): The path to the raw data file. If None, the default train raw file will be used.
                model_parameters (Dict): Optional dictionary of model parameters.
            """
            if raw_file is None:
                raw_file = self.settings.DEFAULT_TRAIN_RAW_FILE
            transformer, df_train, df_eval = self._prepare(raw_file)
            self._train(raw_file, df_train, transformer, model_parameters)

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

    def _prepare(self, raw_file: str) -> Tuple[Pipeline, pd.DataFrame, pd.DataFrame]:
        df_raw = pd_read(raw_file)

        transformer = self._create_transformer()

        X_train, X_test, y_train, y_test = train_test_split(
            df_raw.drop(columns=[self.settings.TARGET_COL]),
            df_raw[self.settings.TARGET_COL],
            test_size=0.2,
            random_state=42,
        )

        # X_train = transformer.fit_transform(X_train)
        # X_test = transformer.transform(X_test)

        return (
            transformer,
            pd.concat([X_train, y_train], axis=1),
            pd.concat([X_test, y_test], axis=1),
        )

    def _train(
        self,
        raw_file: str,
        train_data: pd.DataFrame,
        transformer: Optional[Any],
        model_parameters: Dict = {},
    ) -> str:
        with mlflow.start_run():
            mlflow.sklearn.autolog(
                log_input_examples=True,
                log_datasets=False,
                log_model_signatures=False,
                registered_model_name=self.settings.MODEL_NAME,
            )

            model, _, y_predict = self.train(train_data, transformer, model_parameters)
            reference_file = pd_write_random(
                train_data.assign(prediction=y_predict),
                self.settings.REFERENCE_PATH,
                extension="csv",
            )
            logger.info(f"reference file: {reference_file}")
            # set_signature('mlflow-artifacts:/model', infer_signature(train_data.drop(columns=self.settings.TARGET_COL), y_predict))
            mlflow.log_input(
                mlflow.data.from_pandas(
                    train_data,
                    source=raw_file,
                    targets=self.settings.TARGET_COL,
                    name="training",
                ),
                context="training",
            )
            mlflow.log_input(
                mlflow.data.from_pandas(
                    train_data.assign(prediction=y_predict),
                    source=reference_file,
                    targets=self.settings.TARGET_COL,
                    name="reference",
                ),
                context="training",
            )

            run_id = mlflow.active_run().info.run_id
            return run_id

    def train(
        self, df_train: pd.DataFrame, pipeline: Pipeline, model_parameters: Dict = {}
    ) -> Tuple[Pipeline, pd.DataFrame, np.ndarray]:
        """
        The function takes in a dataframe and a dictionary of model parameters, and returns a trained
        model

        :param df_train: The training dataframe
        :type df_train: pd.DataFrame
        :param model_parameters: Dict
        :type model_parameters: Dict
        :return: The model object
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

