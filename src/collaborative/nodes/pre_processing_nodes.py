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
    raw_path: Path = paths.get_path(paths.DATA_01RAW, source, as_string=False)

    if not raw_path.is_dir():
        catalog.create_raw_dataset(source)

    if isinstance(dataset_name, str):
        _make_raw_file(
            session, source, dataset_name, from_format, to_format, weights, seed
        )
        return [dataset_name]
    else:
        dataset_names: list = []
        for dn in dataset_name:
            _make_raw_file(session, source, dn, from_format, to_format, weights, seed)
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
        as_string=True,  # )
    )
    dataset = session.read.parquet(raw_datset_path)
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

    train, prod = dataset.randomSplit(weights, seed=seed)

    raw_train_filepath = paths.get_path(
        paths.DATA_01RAW,
        source,
        catalog.DatasetType.TRAIN,
        dataset_name,
        suffix=to_format,
        as_string=True,
    )

    raw_prod_filepath = paths.get_path(
        paths.DATA_01RAW,
        source,
        catalog.DatasetType.PROD,
        dataset_name,
        suffix=to_format,
        as_string=True,
    )

    train.write.mode("overwrite").parquet(raw_train_filepath)
    prod.write.mode("overwrite").parquet(raw_prod_filepath)
