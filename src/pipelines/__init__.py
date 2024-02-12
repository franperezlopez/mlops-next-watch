from abc import ABC, abstractmethod
import enum


class Pipelines(enum.Enum):
    TRAINING = "training"
    INFERENCE = "inference"
    MONITOR = "monitor"


class Pipeline(ABC):
    @abstractmethod
    def run(self, **kwargs):
        pass