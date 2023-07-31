import os
from pathlib import Path
from typing import Union

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import rand, split

from conf import catalog, globals, params, paths


def make_raw_datasets(
    session: SparkSession,
    source: str,
    dataset_name: Union[str, list[str]],
    from_format: str,
    to_format: str,
    weights: list[float],
    seed: int,
    split_based_on_column: str = "",
) -> dict[str, dict[str, DataFrame]]:
    raw_path: Path = paths.get_path(paths.DATA_01RAW, source, as_string=False)

    if not raw_path.is_dir():
        catalog.create_raw_dataset(source)

    if isinstance(dataset_name, str):
        _make_raw_file(
            session,
            source,
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
            _make_raw_file(
                session,
                source,
                dn,
                from_format,
                to_format,
                weights,
                seed,
                split_based_on_column,
            )
            dataset_names += [dn]
        return dataset_names


def drop_columns(
    session: SparkSession, source: str, dataset_name: str, dataset_type: str, *columns
):
    raw_datset_path = paths.get_path(
        paths.DATA_01RAW,
        source,
        dataset_type,
        dataset_name,
        suffix=catalog.FileFormat.PARQUET,
        storage=globals.Storage.S3,
        as_string=True,  # )
    )
    dataset = session.read.parquet(raw_datset_path)
    dataset = dataset.drop(*columns)
    processed_dataset_path = paths.get_path(
        paths.DATA_02PROCESSED,
        source,
        dataset_type,
        dataset_name,
        suffix=catalog.FileFormat.PARQUET,
        storage=globals.Storage.S3,
        as_string=True,
    )
    dataset.write.mode("overwrite").parquet(processed_dataset_path)


def _make_raw_file(
    session: SparkSession,
    source: str,
    dataset_name: str,
    from_format: str,
    to_format: str,
    weights: list[float],
    seed: int,
    split_based_on_column: str,
):
    ext_filepath = paths.get_path(
        paths.DATA_01EXTERNAL,
        source,
        dataset_name,
        suffix=from_format,
        as_string=True,
    )

    dataset = session.read.load(
        ext_filepath, format=catalog.FileFormat.CSV[1:], header=True, inferSchema=True
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
        source,
        dataset_name,
        suffix=from_format,
        storage=globals.Storage.S3,
        as_string=True,
        s3_protocol=globals.Protocols.S3,
    )

    raw_train_filepath = paths.get_path(
        paths.DATA_01RAW,
        source,
        catalog.DatasetType.TRAIN,
        dataset_name,
        suffix=to_format,
        storage=globals.Storage.S3,
        as_string=True,
    )

    raw_prod_filepath = paths.get_path(
        paths.DATA_01RAW,
        source,
        catalog.DatasetType.PRODUCTION,
        dataset_name,
        suffix=to_format,
        storage=globals.Storage.S3,
        as_string=True,
    )
    raw_train_filepath_CSV = paths.get_path(
        paths.DATA_01RAW,
        source,
        catalog.DatasetType.TRAIN,
        dataset_name,
        suffix=catalog.FileFormat.CSV,
        storage=globals.Storage.S3,
        as_string=True,
    )

    raw_prod_filepath_CSV = paths.get_path(
        paths.DATA_01RAW,
        source,
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
    # .coalesce(1).write.option("header", True).mode("overwrite").csv(
    #    remote_ext_filepath
    # )
    train.write.mode("overwrite").parquet(raw_train_filepath)
    prod.write.mode("overwrite").parquet(raw_prod_filepath)
