from prometheus_client import Gauge, CollectorRegistry
from typing import Optional

class Monitoring():
    """
    Class for managing Prometheus metrics.
    """

    _initialized = False
    REGISTRY: Optional[CollectorRegistry] = None
    _prom_ground_truth_drift: Optional[Gauge] = None
    _prom_prediction_drift: Optional[Gauge] = None
    _prom_dataset_drift_share: Optional[Gauge] = None

    @classmethod
    def initialize(cls) -> None:
        """
        Initializes the Monitoring class by creating a CollectorRegistry and initializing the Prometheus metrics.
        """
        if not cls._initialized:
            cls.REGISTRY = CollectorRegistry()
            cls._prom_ground_truth_drift = Gauge(
                "ground_truth_drift", "Ground Truth drift", registry=cls.REGISTRY
            )
            cls._prom_prediction_drift = Gauge(
                "prediction_drift", "Prediction Drift", registry=cls.REGISTRY
            )
            cls._prom_dataset_drift_share = Gauge(
                "dataset_drift_share", "Dataset Drift Share", registry=cls.REGISTRY
            )
            cls._initialized = True

    @classmethod
    def create_prometheus_metrics(cls, ground_truth_drift: float, prediction_drift: float, 
                                  dataset_drift_share: float):
        """
        Sets the values of the Prometheus metrics.

        Args:
            ground_truth_drift (float): The value of the ground truth drift metric.
            prediction_drift (float): The value of the prediction drift metric.
            dataset_drift_share (float): The value of the dataset drift share metric.
        """
        cls._prom_ground_truth_drift.set(ground_truth_drift)
        cls._prom_prediction_drift.set(prediction_drift)
        cls._prom_dataset_drift_share.set(dataset_drift_share)
