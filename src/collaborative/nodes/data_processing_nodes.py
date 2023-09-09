import os
from pathlib import Path
from typing import Union

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    current_timestamp,
    monotonically_increasing_id,
    row_number,
)
from pyspark.sql.types import TimestampType
from pyspark.sql.window import Window

from conf import catalog, globals, paths


class DataProcessingNodes:
    def __init__(
        self,
        session: SparkSession,
        source: str,
        storage: str,
    ) -> None:
        self.session = session
        self.source = source
        self.storage = storage

    def make_raw_datasets(
        self,
        dataset_name: Union[str, list[str]],
        from_format: str,
        to_format: str,
        weights: list[float],
        seed: int,
        split_based_on_column: str = "",
    ) -> dict[str, dict[str, DataFrame]]:
        raw_path: Path = paths.get_path(paths.DATA_01RAW, self.source, as_string=False)

        if not raw_path.is_dir():
            catalog.create_raw_dataset(self.source)

        if isinstance(dataset_name, str):
            self._make_raw_file(
                dataset_name,
                from_format,
                to_format,
                weights,
                seed,
                split_based_on_column,
            )
            return [dataset_name]
        else:
            dataset_names: list = []
            for dn in dataset_name:
                self._make_raw_file(
                    dn,
                    from_format,
                    to_format,
                    weights,
                    seed,
                    split_based_on_column,
                )
                dataset_names += [dn]
            return dataset_names

    def drop_columns(self, dataset_name: str, dataset_type: str, *columns):
        raw_datset_path = paths.get_path(
            paths.DATA_01RAW,
            self.source,
            dataset_type,
            dataset_name,
            suffix=catalog.FileFormat.PARQUET,
            storage=globals.Storage.S3,
            as_string=True,
        )
        dataset = self.session.read.parquet(raw_datset_path)
        dataset = dataset.drop(*columns)

        processed_dataset_path = paths.get_path(
            paths.DATA_02PROCESSED,
            self.source,
            dataset_type,
            dataset_name,
            suffix=catalog.FileFormat.PARQUET,
            storage=globals.Storage.S3,
            as_string=True,
        )
        dataset.write.mode("overwrite").parquet(processed_dataset_path)

    def _add_datetime(self, dataset: DataFrame):
        dates = pd.date_range(
            start="2017-01-01", end="2017-01-04", periods=dataset.count()
        )
        datetimes = [date.to_pydatetime() for date in dates]
        dates = self.session.createDataFrame(datetimes, TimestampType())
        dataset = dataset.withColumn(
            "row_index", row_number().over(Window.orderBy(monotonically_increasing_id()))
        )
        dates = dates.withColumn(
            "row_index", row_number().over(Window.orderBy(monotonically_increasing_id()))
        )
        dataset = dataset.join(dates, on=["row_index"]).drop("row_index")
        dataset = dataset.withColumnRenamed("value", "datetime")
        return dataset

    def _make_raw_file(
        self,
        dataset_name: str,
        from_format: str,
        to_format: str,
        weights: list[float],
        seed: int,
        split_based_on_column: str,
    ):
        ext_filepath = paths.get_path(
            paths.DATA_01EXTERNAL,
            self.source,
            dataset_name,
            suffix=from_format,
            as_string=True,
        )
        dataset = self.session.read.load(
            ext_filepath, format=catalog.FileFormat.CSV[1:], header=True, inferSchema=True
        )
        dataset = dataset.withColumn(
            "datetime", current_timestamp()
        )  

        if split_based_on_column == "":
            train, prod = dataset.randomSplit(weights, seed=seed)
        else:
            unique_items = dataset[[split_based_on_column]].distinct()
            train_items, prod_items = unique_items.randomSplit(weights, seed=seed)
            train = dataset.join(train_items, on=split_based_on_column)
            prod = dataset.join(prod_items, on=split_based_on_column)

        remote_ext_filepath = paths.get_path(
            paths.DATA_01EXTERNAL,
            self.source,
            dataset_name,
            suffix=from_format,
            storage=globals.Storage.S3,
            as_string=True,
            s3_protocol=globals.Protocols.S3,
        )

        raw_train_filepath = paths.get_path(
            paths.DATA_01RAW,
            self.source,
            catalog.DatasetType.TRAIN,
            dataset_name,
            suffix=to_format,
            storage=globals.Storage.S3,
            as_string=True,
        )

        raw_prod_filepath = paths.get_path(
            paths.DATA_01RAW,
            self.source,
            catalog.DatasetType.PRODUCTION,
            dataset_name,
            suffix=to_format,
            storage=globals.Storage.S3,
            as_string=True,
        )
        raw_train_filepath_CSV = paths.get_path(
            paths.DATA_01RAW,
            self.source,
            catalog.DatasetType.TRAIN,
            dataset_name,
            suffix=catalog.FileFormat.CSV,
            storage=globals.Storage.S3,
            as_string=True,
        )

        raw_prod_filepath_CSV = paths.get_path(
            paths.DATA_01RAW,
            self.source,
            catalog.DatasetType.PRODUCTION,
            dataset_name,
            suffix=catalog.FileFormat.CSV,
            storage=globals.Storage.S3,
            as_string=True,
        )
        storage_options = {
            "key": os.getenv("AWS_ACCESS_KEY_ID"),
            "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
        }
        dataset.toPandas().to_csv(
            remote_ext_filepath,
            index=False,
            storage_options=storage_options,
        )
        train.toPandas().to_csv(
            raw_train_filepath_CSV,
            index=False,
            storage_options=storage_options,
        )
        prod.toPandas().to_csv(
            raw_prod_filepath_CSV,
            index=False,
            storage_options=storage_options,
        )
        train.write.mode("overwrite").parquet(raw_train_filepath)
        prod.write.mode("overwrite").parquet(raw_prod_filepath)
