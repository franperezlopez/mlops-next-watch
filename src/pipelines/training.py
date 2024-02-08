from typing import Any, Dict
import argparse
import os
import sys
from pathlib import Path

from joblib import dump
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
import mlflow
import mlflow.sklearn

from src.model.settings import TARGET_COL


def train(df_train: pd.DataFrame, model_parameters: Dict = {}) -> Any:
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
    y_train = df_train[TARGET_COL]
    X_train = df_train.drop(columns=TARGET_COL)

    # Train a Linear Regression Model with the train set
    model = RandomForestRegressor(random_state=0, **model_parameters)
    model.fit(X_train, y_train)

    # Train metrics are automatically logged when using mlflow.autolog
    model.score(X_train, y_train)

    return model


def main_prepare(raw_data: Path, prepared_data: Path, artifacts_data: Path, 
         raw_file: str = DEFAULT_TRAIN_RAW_FILE):
    from src.model.prepare import create_transformer, prepare_and_save_splits
    from src.model.settings import TRANSFORMER_FILE
    from src.services.appinsights import AppInsightsLogger
    from src.services.utils import pd_read, pd_write

    logger.info(f"raw data folder: {raw_data}")
    logger.info(f"raw data files: {os.listdir(raw_data)}")
    logger.info(f"prepared data folder: {prepared_data}")

    df_raw = pd_read(raw_data / raw_file)

    transformer = create_transformer()

    transformer = prepare_and_save_splits(df_raw, transformer, prepared_data)

    # Save transformer (must be done after t)
    if transformer is not None:
        dump(transformer, artifacts_data / TRANSFORMER_FILE)


def main_train(prepared_data: Path, artifacts_data: Path, nodeploy: bool = False, model_parameters: Dict = {}) -> str:
    from src.model.settings import MODEL_NAME, SPLIT_TEST_FILE, SPLIT_TRAIN_FILE, TRANSFORMER_FILE
    from src.model.train import train
    from src.services.mlflow import evaluate_run_id_model_on_data, register_model_using_improve_metric_policy
    from src.services.utils import pd_read

    deploy = not nodeploy
    logger.info(f"prepared_data folder: {prepared_data}")
    logger.info(f"prepared_data files: {os.listdir(prepared_data)}")
    logger.info(f"artifacts_data folder: {artifacts_data}")
    logger.info(f"deploy: {deploy}")

    update_experiment_name("train_model")
    # run for training
    with mlflow.start_run():
        # Enable auto logging (only for scikit-learn)
        mlflow.sklearn.autolog(log_models=True)

        # Load the train data
        train_data = pd_read(prepared_data / SPLIT_TRAIN_FILE)
        model = train(train_data, model_parameters)

        # Log transformer
        if (artifacts_data / TRANSFORMER_FILE).exists():
            logger.info("logging transformer")
            mlflow.log_artifact(str(artifacts_data / TRANSFORMER_FILE), "model")

        run_id = mlflow.active_run().info.run_id

    # Run for evaluation (same run_id as training)
    evaluate_run_id_model_on_data(run_id, run_id, prepared_data / SPLIT_TEST_FILE, MODEL_NAME)

    # Register model (mandatory policy)
    if deploy:
        register_model_using_improve_metric_policy(MODEL_NAME, run_id)

    return run_id