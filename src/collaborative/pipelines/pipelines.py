from pyspark.sql import DataFrame, SparkSession

from collaborative.nodes import data_science_nodes, pre_processing_nodes
from conf import catalog, globals, params, paths


class Pipelines:
    def __init__(self, session: SparkSession, source: str) -> None:
        self.session = session
        self.source = source

    def data_engineering(
        self,
        from_format: str = catalog.FileFormat.CSV,
        to_format: str = catalog.FileFormat.PARQUET,
        weights: list[float] = [params.RAW, 1 - params.RAW],
        seed: int = params.SEED,
    ):
        datasets = pre_processing_nodes.make_raw_datasets(
            self.session,
            self.source,
            [
                catalog.Datasets.RATINGS,
                catalog.Datasets.MOVIES,
                catalog.Datasets.TAGS,
                catalog.Datasets.LINKS,
            ],
            from_format,
            to_format,
            weights,
            seed,
        )

        train = pre_processing_nodes.drop_columns(
            self.source,
            datasets[catalog.Datasets.RATINGS][catalog.DatasetType.TRAIN],
            catalog.Datasets.RATINGS,
            catalog.DatasetType.TRAIN,
            *globals.DROP_COLUMNS,
        )

        serve = pre_processing_nodes.drop_columns(
            self.source,
            datasets[catalog.Datasets.RATINGS][catalog.DatasetType.RAW],
            catalog.Datasets.RATINGS,
            catalog.DatasetType.RAW,
            *globals.DROP_COLUMNS,
        )

        return train, serve

    def data_science(self, dataset: DataFrame = None):
        if dataset is None:
            dataset = self.session.read.parquet(
                paths.get_path(
                    paths.DATA_03PROCESSED,
                    self.source,
                    catalog.DatasetType.TRAIN,
                    catalog.Datasets.RATINGS,
                    suffix=catalog.FileFormat.PARQUET,
                    as_string=True,
                )
            )
        train, val = data_science_nodes.split_train_test(dataset)
        best_trial = data_science_nodes.hyperparam_opt_als(train, val)

        print(best_trial)
