from prometheus_client import Gauge, CollectorRegistry
from typing import Optional

class Monitoring():
    _initialized = False
    REGISTRY: Optional[CollectorRegistry] = None
    _prom_ground_truth_drift: Optional[Gauge] = None
    _prom_prediction_drift: Optional[Gauge] = None
    _prom_dataset_drift_share: Optional[Gauge] = None

    @classmethod
    def initialize(cls) -> None:
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

        cls._prom_ground_truth_drift.set(ground_truth_drift)
        cls._prom_prediction_drift.set(prediction_drift)
        cls._prom_dataset_drift_share.set(dataset_drift_share)
