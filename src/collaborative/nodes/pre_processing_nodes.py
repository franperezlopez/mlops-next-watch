from pathlib import Path

from pyspark.sql import DataFrame, SparkSession

from conf import catalog, params, paths


def make_raw_datasets(
    session: SparkSession,
    source: str,
    dataset_name: str | list[str],
    from_format: str,
    to_format: str,
    weights: list[float],
    seed: int,
) -> dict[str, dict[str, DataFrame]]:
    raw_path: Path = paths.get_path(paths.DATA_01RAW, source)

    if not raw_path.is_dir():
        catalog.create_raw_dataset(source)

    returns: dict = {}

    if isinstance(dataset_name, str):
        train, serve = _make_raw_file(
            session, source, dataset_name, from_format, to_format, weights, seed
        )
        returns[dataset_name] = {
            catalog.DatasetType.TRAIN: train,
            catalog.DatasetType.RAW: serve,
        }
        return returns
    else:
        for dn in dataset_name:
            train, serve = _make_raw_file(
                session, source, dn, from_format, to_format, weights, seed
            )
            returns[dn] = {
                catalog.DatasetType.TRAIN: train,
                catalog.DatasetType.RAW: serve,
            }
        return returns


def drop_columns(
    source: str, dataset: DataFrame, dataset_name: str, dataset_type: str, *columns
):
    dataset = dataset.drop(*columns)
    processed_dataset_path = paths.get_path(
        paths.DATA_03PROCESSED,
        source,
        dataset_type,
        dataset_name,
        suffix=catalog.FileFormat.PARQUET,
        as_string=True,
    )
    dataset.write.mode("overwrite").parquet(processed_dataset_path)
    return dataset


def _make_raw_file(
    session: SparkSession,
    source: str,
    dataset_name: str,
    from_format: str,
    to_format: str,
    weights: list[float],
    seed: int,
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

    train, serve = dataset.randomSplit(weights, seed=seed)

    raw_train_filepath = paths.get_path(
        paths.DATA_01RAW,
        source,
        catalog.DatasetType.TRAIN,
        dataset_name,
        suffix=to_format,
        as_string=True,
    )

    raw_serve_filepath = paths.get_path(
        paths.DATA_01RAW,
        source,
        catalog.DatasetType.RAW,
        dataset_name,
        suffix=to_format,
        as_string=True,
    )

    train.write.mode("overwrite").parquet(raw_train_filepath)
    serve.write.mode("overwrite").parquet(raw_serve_filepath)

    return train, serve
