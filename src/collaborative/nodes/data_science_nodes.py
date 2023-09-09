from hyperopt import Trials, fmin, hp, tpe
from pyspark.sql import SparkSession

from collaborative.models import als_model
from conf import catalog, params, paths
from utils.experiment_tracking import MLflowHandler


class DataScienceNodes:
    def __init__(
        self,
        session: SparkSession,
        source: str,
        dataset: str,
        file_suffix: str,
        storage: str,
    ) -> None:
        self.session = session
        self.source = source
        self.dataset = dataset  # catalog.Datasets.RATINGS
        self.file_suffix = file_suffix  # catalog.FileFormat.PARQUET
        self.storage = storage  # globals.Storage.S3

        self.processed_data_path = paths.get_path(
            paths.DATA_02PROCESSED,
            self.source,
            catalog.DatasetType.TRAIN,
            self.dataset,
            suffix=self.file_suffix,
            storage=self.storage,
            as_string=True,
        )
        self.train_data_path = paths.get_path(
            paths.DATA_03TRAIN,
            self.source,
            catalog.DatasetType.TRAIN,
            self.dataset,
            suffix=self.file_suffix,
            storage=self.storage,
            as_string=True,
        )
        self.test_data_path = paths.get_path(
            paths.DATA_03TRAIN,
            self.source,
            catalog.DatasetType.TEST,
            self.dataset,
            suffix=self.file_suffix,
            storage=self.storage,
            as_string=True,
        )

    def _train_als(self, train, val):
        """Where the train is taken..."""

        @MLflowHandler.log_train
        def train_wrapper(params):

            model = als_model.get_model(params)
            evaluator = als_model.get_model_evaluator()

            model = model.fit(train)

            predictions = model.transform(val)

            metric = evaluator.evaluate(predictions)

            return metric, model, predictions.toJSON().collect()

        return train_wrapper

    def split_train_test(self):
        """
        Splits the training data into train and test sets
        """
        dataset = self.session.read.parquet(self.processed_data_path)
        train, test = dataset.randomSplit([params.TRAIN, 1 - params.TRAIN], seed=42)
        train.write.mode("overwrite").parquet(self.train_data_path)
        test.write.mode("overwrite").parquet(self.test_data_path)

    def hyperparam_opt_als(self):
        """Performs hyper-parameter optimization

        Args:
            train (DataFrame): A `Spark` `DataFrame` containing the training set

        """

        rank = params.ALS.RANK
        reg_param = params.ALS.REG_INTERVAL
        cold_start_strat = params.ALS.COLD_START_STRATEGY
        max_iter = params.ALS.MAX_ITER

        space_search = {
            "rank": hp.choice("rank", rank),
            "reg_param": hp.uniform("reg_param", reg_param[0], reg_param[1]),
            "cold_start_strategy": hp.choice("cold_start_strategy", cold_start_strat),
            "max_iter": hp.choice("max_iter", max_iter),
        }

        train = self.session.read.parquet(self.train_data_path)
        val = self.session.read.parquet(self.test_data_path)

        # FROM `hyperopt` docs: Do not use the SparkTrials class with MLlib. SparkTrials is designed to distribute trials for algorithms that are not themselves distributed.
        trials = Trials()

        self._run_hyperparam_opt(train, val, space_search, trials)

    @MLflowHandler.log_hyperparam_opt
    def _run_hyperparam_opt(self, train, val, space_search, trials):
        # Start Hyper-Parameter Optimization
        best = fmin(
            fn=self._train_als(train=train, val=val),
            space=space_search,
            algo=tpe.suggest,
            max_evals=5,
            trials=trials,
        )

        return best
