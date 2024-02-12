from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, computed_field
from typing import List
from pathlib import Path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    TARGET_COL: str = Field("cost")
    PREDICTION_COL: str = Field("prediction")

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

    @computed_field
    @property
    def TRAIN_COLS(self) -> List[str]:
        return self.NUMERIC_COLS + self.CAT_NOM_COLS + self.CAT_ORD_COLS
    
    MODEL_NAME: str = Field("taxi_fare_model")

    MONITOR_PATH: Path = Field(Path("../data/03-predictions"))
    REFERENCE_PATH: Path = Field(Path("../data/02-processed"))

@lru_cache
def get_settings():
    return Settings()