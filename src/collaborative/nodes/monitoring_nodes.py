import ast
import datetime
import os

import pandas as pd
from dotenv import load_dotenv
from evidently import ColumnMapping
from evidently.metric_preset import (
    DataDriftPreset,
    RegressionPreset,
    TargetDriftPreset,
)
from evidently.report import Report
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from conf import catalog, globals, params, paths
from utils.experiment_tracking import MLflowHandler
from utils.psycopg_handler import PsycopgHandler


class MonitoringNodes:
    def __init__(self) -> None:
        env_path = paths.get_path(
            paths.ENV,
            storage=globals.Storage.DOCKER,
        )
        load_dotenv(env_path)
        self.prom_registry = CollectorRegistry()
        self.ground_truth_drift = 0
        self.prediction_drift = 0
        self.dataset_drift_share = 0
        self.dataset_n_drifted_columns = 0

    def get_training_data(self, artifact_path: str):
        train_predictions = MLflowHandler.get_artifacts_from_model(
            model_name=globals.MLflow.ALS_REGISTERED_MODEL_NAME,
            stage="Production",
            artifact_path=artifact_path,
            as_dict=True,
        )
        train_predictions = [ast.literal_eval(p) for p in train_predictions]
        train_predictions_df = pd.DataFrame(train_predictions)
        train_predictions_df["datetime"] = pd.to_datetime(
            train_predictions_df["datetime"]
        ).dt.tz_localize(None)
        return train_predictions_df

    def get_production_data(
        self,
    ):
        remote_ratings_path = paths.get_path(
            paths.DATA_01EXTERNAL,
            catalog.Sources.MOVIELENS,
            catalog.Datasets.RATINGS,
            suffix=catalog.FileFormat.CSV,
            storage=globals.Storage.S3,
            as_string=True,
            s3_protocol=globals.Protocols.S3,
        )
        most_recent_ratings_csv = pd.read_csv(remote_ratings_path)
        psycopg = PsycopgHandler(
            os.getenv("POSTGRES_USER"),
            os.getenv("POSTGRES_PASSWORD"),
            os.getenv("POSTGRES_IP"),
            os.getenv("POSTGRES_PORT"),
            os.getenv("POSTGRES_APP_DATABASE"),
        )
        prod_predictions = pd.DataFrame(
            psycopg.read_db("SELECT * FROM recommendations"),
            columns=["index", "userId", "movieId", "prediction", "rank", "datatime"],
        )
        prod_data = (
            prod_predictions.merge(
                most_recent_ratings_csv,
                on=["userId", "movieId"],
                how="left",
                suffixes=["_pred", "_rating"],
            )
            .drop(["index", "rank", "timestamp"], axis=1)
            .rename(columns={"datetime_pred": "datetime"})
        )

        prod_data["datetime"] = pd.to_datetime(prod_data["datetime"]).dt.tz_localize(None)
        return prod_data

    def report_metrics(self, prod_data, train_data):
        column_mapping = ColumnMapping()
        column_mapping.prediction = "prediction"
        column_mapping.target = "rating"
        column_mapping.numerical_features = ["movieId"]
        column_mapping.id = "userId"
        column_mapping.datetime = (
            None  # disable datetime because we are doing batch monitoring...
        )

        report_args = {
            "current_data": prod_data,
            "reference_data": train_data,
            "column_mapping": column_mapping,
        }

        self.model_report = Report(
            metrics=[RegressionPreset()],
            timestamp=datetime.datetime.now(),
        )
        self.drift_report = Report(
            metrics=[
                TargetDriftPreset(stattest_threshold=params.Monitoring.TARGET_DRIFT_THR),
                DataDriftPreset(stattest_threshold=params.Monitoring.DATA_DRIFT_THR),
            ],
            timestamp=datetime.datetime.now(),
        )
        self.model_report.run(**report_args)
        self.drift_report.run(**report_args)

        time_format = "%Y-%m-%d_%Hh%Mm%Ss"
        model_report_filename = "model_report_" + datetime.datetime.now().strftime(time_format)
        drift_report_filename = "drift_report_" +  datetime.datetime.now().strftime(time_format)

        self.model_report.save_html(
            paths.get_path(
                paths.DATA_04MONITORING, model_report_filename, suffix=".html", as_string=True
            )
        )
        self.drift_report.save_html(
            paths.get_path(
                paths.DATA_04MONITORING, drift_report_filename, suffix=".html", as_string=True
            )
        )

        self.drift_report_dict = self.drift_report.as_dict()

    def get_metrics_from_drift_report(self):  # drift_report only

        for metric in self.drift_report_dict["metrics"]:
            metric_result = metric["result"]
            if metric["metric"] == "ColumnDriftMetric":
                if metric_result["column_name"] == "rating":
                    self.ground_truth_drift = metric_result["drift_score"]
                elif metric_result["column_name"] == "prediction":
                    self.prediction_drift = metric_result["drift_score"]
            elif metric["metric"] == "DatasetDriftMetric":
                self.dataset_drift_share = metric_result["drift_share"]
                self.dataset_n_drifted_columns = metric_result[
                    "number_of_drifted_columns"
                ]

    def send_to_prometheus(self):

        prom_ground_truth_drift = Gauge(
            "ground_truth_drift", "Ground Truth drift", registry=self.prom_registry
        )
        prom_prediction_drift = Gauge(
            "prediction_drift", "Prediction Drift", registry=self.prom_registry
        )
        prom_dataset_drift_share = Gauge(
            "dataset_drift_share", "Dataset Drift Share", registry=self.prom_registry
        )
        prom_dataset_n_drifted_columns = Gauge(
            "dataset_n_drifted_columns",
            "Number of Dataset Drifted Columns",
            registry=self.prom_registry,
        )

        prom_ground_truth_drift.set(self.ground_truth_drift)
        prom_prediction_drift.set(self.prediction_drift)
        prom_dataset_drift_share.set(self.dataset_drift_share)
        prom_dataset_n_drifted_columns.set(self.dataset_n_drifted_columns)

        push_to_gateway(
            f"pushgateway:{os.getenv('PUSHGATEWAY_PORT')}",
            job="evidently_metrics",
            registry=self.prom_registry,
        )
