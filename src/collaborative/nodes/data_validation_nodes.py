from mlflow.utils.logging_utils import sys
from pyspark.sql.functions import max, min

from conf import paths


class DataValidation:
    def __init__(self, session, source, storage) -> None:
        self.session = session
        self.source = source
        self.storage = storage

    def validate_ratings(self, dataset: str, file_suffix: str):

        try:
            self.processed_data_path = paths.get_path(
                paths.DATA_01EXTERNAL,
                self.source,
                dataset,
                suffix=file_suffix,
                storage=self.storage,
                as_string=True,
            )
            data = self.session.read.csv(
                self.processed_data_path,
                sep=",",
                header=True,
                inferSchema=True,
            )
            maxv, minv = data.agg(max("rating"), min("rating")).first().asDict().values()
            assert maxv <= 5 and minv >= 0, "Ratings are NOT in range [0, 5]"

        except Exception as e:
            print(f"Exception {e} occurred, exiting...")
            sys.exit(1)
