import pandas as pd
import datetime
from evidently import ColumnMapping
from evidently.metric_preset import (
    DataDriftPreset,
    RegressionPreset,
    TargetDriftPreset,
)
from evidently.options.data_drift import DataDriftOptions
from evidently.report import Report
import mlflow
from loguru import logger

from src.pipelines import Pipeline
from src.services.mlflow import Mlflow
from src.settings import Settings
from src.utils.pandas import pd_read
from src.services.prometheus import Monitoring


class Monitor(Pipeline):
    def __init__(self, settings: Settings):
        self.settings: Settings = settings
        Monitoring.initialize()

    def run(self):
            """
            Runs the monitoring logic.

            Retrieves the current data, gets the reference data for each runid,
            builds a drift report, creates dataset sources, updates the run with monitoring,
            and creates Prometheus metrics.

            Returns:
                None
            """
            # Monitoring logic here
            df_current = self._get_current_data().dropna()
            logger.info(f"Monitoring {df_current.filename.nunique()} files")
            logger.info(f"Current data shape {df_current.shape}")
            runids = df_current.runid.unique()
            df_reference = self._get_reference_data(runids)
            for runid in runids:
                logger.info(f"Reporting runid {runid}")
                drift_report = self._build_report(
                    df_current[df_current.runid == runid],
                    df_reference[df_reference.runid == runid],
                )
                dataset_sources = self._create_dataset_sources(df_current, runid)
                prometheus_metrics = self._update_run_with_monitoring(runid, dataset_sources, drift_report)
                Monitoring.create_prometheus_metrics(*prometheus_metrics)

    def _create_dataset_sources(self, df_current, runid):
        dataset_sources = [
                    mlflow.data.from_pandas(
                        df_current[(df_current.runid == runid) & (df_current.filename == filename)],
                        source=filename,
                        targets=self.settings.TARGET_COL,
                        name=f"reference_{filename}",
                    )
                    for filename in df_current[df_current.runid == runid].filename.unique()
                ]
        
        return dataset_sources

    def _get_current_data(self) -> pd.DataFrame:
        # Get current data
        file_contents = []
        for file in self.settings.MONITOR_PATH.iterdir():
            file_contents.append(pd_read(file).assign(filename=file.name))

        return pd.concat(file_contents)

    def _get_reference_data(self, runids: list[str]) -> pd.DataFrame:
        # Get reference data
        file_contents = []
        for runid in runids:
            # Get dataset from MLflow
            dataset_path = Mlflow.get_dataset_path(runid)
            # Load dataset
            file_contents.append(pd_read(dataset_path).assign(runid=runid))

        return pd.concat(file_contents)

    def _build_report(
        self, df_current: pd.DataFrame, df_reference: pd.DataFrame
    ) -> Report:
        column_mapping = ColumnMapping(
            target=self.settings.TARGET_COL,
            categorical_features=self.settings.CAT_NOM_COLS,
            numerical_features=self.settings.NUMERIC_COLS,
        )
        report_args = {
            "current_data": df_current,
            "reference_data": df_reference,
            "column_mapping": column_mapping,
        }

        # model_report = Report(
        #     metrics=[RegressionPreset()],
        #     timestamp=datetime.datetime.now(),
        # )
        drift_report = Report(
            metrics=[
                TargetDriftPreset(stattest_threshold=0.1),
                DataDriftPreset(stattest_threshold=0.1),  # detect features drift
            ],
            timestamp=datetime.datetime.now(),
        )
        # model_report.run(**report_args)
        drift_report.run(**report_args)

        return drift_report

    def _detect_dataset_drift(self, data_drift_report, get_ratio=False):
        """
        Returns True if Data Drift is detected, else returns False.
        If get_ratio is True, returns the share of drifted features.
        The Data Drift detection depends on the confidence level and the threshold.
        For each individual feature Data Drift is detected with the selected confidence (default value is 0.95).
        Data Drift for the dataset is detected if share of the drifted features is above the selected threshold (default value is 0.5).
        """

        # data_drift_report = Report(metrics=[DataDriftPreset()])
        # data_drift_report.run(reference_data=reference, current_data=production, column_mapping=column_mapping)
        report = data_drift_report.as_dict()

        for metric in report["metrics"]:
            if metric["metric"] == "DatasetDriftMetric":
                if get_ratio:
                    drift = metric["result"]["drift_share"]
                else:
                    drift = metric["result"]["dataset_drift"]

        return drift  # , report["metrics"][0]["result"]["number_of_drifted_columns"]

    def _detect_features_drift(
        self, data_drift_report
    ) -> dict[str, tuple[float, bool]]:
        """
        Returns True if Data Drift is detected, else returns False.
        If get_scores is True, returns scores value (like p-value) for each feature.
        The Data Drift detection depends on the confidence level and the threshold.
        For each individual feature Data Drift is detected with the selected confidence (default value is 0.95).
        """
        report = data_drift_report.as_dict()
        for metric in report["metrics"]:
            if metric["metric"] == "DataDriftTable":
                drifts = {
                    feature: (
                        metric["result"]["drift_by_columns"][feature]["drift_score"],
                        metric["result"]["drift_by_columns"][feature]["drift_detected"],
                    )
                    for feature in self.settings.NUMERIC_COLS
                    + self.settings.CAT_NOM_COLS
                    + [self.settings.TARGET_COL, self.settings.PREDICTION_COL]
                }
                return drifts

    def _update_run_with_monitoring(self, runid: str, dataset_sources: list[str], drift_report: Report) -> tuple[float, float, float]:
        # Update the run with monitoring results
        with mlflow.start_run(run_id=runid):
            with mlflow.start_run(nested=True):
                # Log the dataset sources
                for dataset in dataset_sources:
                    mlflow.log_input(dataset, context="monitoring")

                # Log the drift report
                features_drift = self._detect_features_drift(drift_report)
                for feature, (score, detected) in features_drift.items():
                    mlflow.log_metric(feature, score)
                    # mlflow.log_metric(f"{feature}_drift_detected", detected)
                dataset_drift = self._detect_dataset_drift(drift_report, get_ratio=True)
                mlflow.log_metric("dataset_drift_share", dataset_drift)

                return (
                    features_drift[self.settings.TARGET_COL][0],
                    features_drift[self.settings.PREDICTION_COL][0],
                    dataset_drift,
                )
