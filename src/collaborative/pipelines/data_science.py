import subprocess

from pyspark.sql import SparkSession

from collaborative.nodes.data_science_nodes import DataScienceNodes
from conf import catalog, globals, params, paths


def run(
    source: str = catalog.Sources.MOVIELENS,
    dataset: str = catalog.Datasets.RATINGS,
    file_suffix: str = catalog.FileFormat.PARQUET,
    storage: str = globals.Storage.S3,
):
    session = (
        SparkSession.builder.appName("collab-ds")
        .config(map=params.Spark.spark_config)
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("INFO")

    data_science_nodes = DataScienceNodes(session, source, dataset, file_suffix, storage)
    data_science_nodes.split_train_test()
    data_science_nodes.hyperparam_opt_als()
    try:
        sparkml_tmp_dir = paths.get_path(
            paths.SPARKML_TMP_DIR, storage=globals.Storage.DOCKER, as_string=True
        )
        subprocess.check_call(["chmod", "-R", "777", sparkml_tmp_dir])
    except subprocess.CalledProcessError as e:
        print(f"Error granting permissions: {e}")
