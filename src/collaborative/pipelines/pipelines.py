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
        pre_processing_nodes.make_raw_datasets(
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

        pre_processing_nodes.drop_columns(
            self.session,
            self.source,
            catalog.Datasets.RATINGS,
            catalog.DatasetType.TRAIN,
            *globals.DROP_COLUMNS,
        )

        pre_processing_nodes.drop_columns(
            self.session,
            self.source,
            catalog.Datasets.RATINGS,
            catalog.DatasetType.PROD,
            *globals.DROP_COLUMNS,
        )

    def data_science(self):
        data_science_nodes.split_train_test(self.session, self.source)
        data_science_nodes.hyperparam_opt_als(self.session, self.source)
