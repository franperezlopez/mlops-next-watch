from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, SecretStr, computed_field
from typing import List, Dict, Any, Optional, Union
import enum

class Pipelines(enum.Enum):
    TRAINING = "training"
    INFERENCE = "inference"
    MONITOR = "monitodr"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    TARGET_COL: str = Field("cost")

    NUMERIC_COLS: List[str] = Field([
        "distance",
        "dropoff_latitude",
        "dropoff_longitude",
        "passengers",
        "pickup_latitude",
        "pickup_longitude",
        "pickup_weekday",
        "pickup_month",
        "pickup_monthday",
        "pickup_hour",
        "pickup_minute",
        "pickup_second",
        "dropoff_weekday",
        "dropoff_month",
        "dropoff_monthday",
        "dropoff_hour",
        "dropoff_minute",
        "dropoff_second",
    ])

    CAT_NOM_COLS: List[str] = Field(["store_forward", "vendor"])

    CAT_ORD_COLS: List[str] = Field([])

    @computed_field
    @property
    def TRAIN_COLS(self) -> List[str]:
        return self.NUMERIC_COLS + self.CAT_NOM_COLS + self.CAT_ORD_COLS
    
    SPLIT_TRAIN_FILE: str = Field("train.parquet")
    SPLIT_VAL_FILE: str = Field("val.parquet")
    SPLIT_TEST_FILE: str = Field("test.parquet")

    TRAIN_MODEL_FILE: str = Field("model.joblib")
    TRANSFORMER_FILE: str = Field("transformer.joblib")
    # PREDICT_STAGE_FILE = "predict.parquet"
    # EVAL_STAGE_FILE = "eval.parquet"

    MODEL_NAME = "taxi_fare_model"

    # def _get_metric_name(model_name=MODEL_NAME):
    #     return f"r2_score_on_data_{model_name}_test"

    # METRIC = _get_metric_name()

    # METRIC_IMPROVEMENT_THRESHOLD = 5. # metric improvement percentage threshold

@lru_cache
def get_settings():
    return Settings()