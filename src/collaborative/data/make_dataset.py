from pyspark.sql import SparkSession

from conf import catalog


class MakeDataset:
    def __init__(self, spark_session: SparkSession, source: str):
        self.spark_session = spark_session
        self.source = source

    def make_raw_datasets(
        self,
        filename: str | list[str],
        from_format: str = catalog.FileFormat.CSV,
        to_format: str = catalog.FileFormat.PARQUET,
        weights: list[float] = [0.20, 0.80],
        seed: int = 42,
    ):
        raw_path = catalog.get_dataset_path(catalog.Paths.DATA_01RAW, self.source)

        if not raw_path.exists() and raw_path.is_dir():
            catalog.create_raw_dataset(self.source)

        if isinstance(filename, str):
            self._make_raw_file(filename, from_format, to_format, weights, seed)
        else:
            for f in filename:
                self._make_raw_file(f, from_format, to_format, weights, seed)

    def _make_raw_file(
        self,
        filename: str,
        from_format: str,
        to_format: str,
        weights: list[float],
        seed: int,
    ):
        ext_filepath = catalog.get_dataset_path(
            catalog.Paths.DATA_01EXTERNAL,
            self.source,
            filename,
            suffix=from_format,
            as_string=True,
        )

        dataset = self.spark_session.read.load(
            ext_filepath, format=catalog.FileFormat.CSV[1:], header=True, inferSchema=True
        )

        train, serve = dataset.randomSplit(weights, seed=seed)

        raw_train_filepath = catalog.get_dataset_path(
            catalog.Paths.DATA_01RAW,
            self.source,
            catalog.DatasetType.TRAIN,
            filename,
            suffix=to_format,
            as_string=True,
        )

        raw_serve_filepath = catalog.get_dataset_path(
            catalog.Paths.DATA_01RAW,
            self.source,
            catalog.DatasetType.SERVE,
            filename,
            suffix=to_format,
            as_string=True,
        )

        train.write.mode("overwrite").parquet(raw_train_filepath)
        serve.write.mode("overwrite").parquet(raw_serve_filepath)
