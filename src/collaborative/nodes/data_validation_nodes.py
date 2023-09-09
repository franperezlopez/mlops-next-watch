from pyspark.sql.functions import max, min

from conf import catalog, paths


class DataValidation:
    def __init__(self, session, source, storage) -> None:
        self.session = session
        self.source = source
        self.storage = storage

    def validate_ratings(self, dataset: str, file_suffix: str):

        self.processed_data_path = paths.get_path(
            paths.DATA_02PROCESSED,
            self.source,
            catalog.DatasetType.TRAIN,
            dataset,
            suffix=file_suffix,
            storage=self.storage,
            as_string=True,
        )
        data = self.session.read.parquet(self.processed_data_path)

        maxv, minv = data.agg(max("rating"), min("rating")).first().asDict().values()
        assert maxv <= 5 and minv >= 0, "Ratings are NOT in range [0, 5]"
