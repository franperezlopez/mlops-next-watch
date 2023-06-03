import logging
import os

import click
import mlflow
from dotenv import load_dotenv
from pyspark.sql import SparkSession

from collaborative.pipelines.pipelines import Pipelines
from conf import catalog, globals


@click.command()
@click.option(
    "-p",
    "--pipelines",
    "pipelines_to_run",
    default=[globals.Pipelines.DATA_SCIENCE],
    type=str,
    multiple=True,
    help="Set Pipelines to run",
)
@click.option(
    "-e",
    "--experiment",
    "experiment_name",
    default=globals.MLflow.EXPERIMENT_NAME,
    type=str,
    nargs=1,
    help="Set Experiment name",
)
def main(pipelines_to_run: str, experiment_name: str):
    """Main Program

    Args:
        pipelines_to_run (str): pipeline names to be run
        experiment_name (str): set an experiment name
    """
    logger = logging.getLogger(__name__)
    # logging.getLogger("mlflow").setLevel(logging.DEBUG)
    # logging.getLogger("pyspark").setLevel(logging.DEBUG)
    logger.info("Running Next-Watch")

    spark = (
        SparkSession.builder.appName("Next Watch ML")
        .master("local[3]")
        # .config("spark.executor.memory", "3g")
        .config("spark.driver.maxResultSize", "96g")
        .config("spark.driver.memory", "96g")
        .config("spark.executor.memory", "8g")
        .config("packages", "org.apache.hadoop:hadoop-aws:3.3.4")
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
        )
        .config(
            "spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem"
        )  # mlflow doesn't support s3a
        .config("spark.hadoop.fs.s3.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .getOrCreate()
    )

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(experiment_name)

    mlflow.spark.autolog(disable=True)

    pipelines = Pipelines(spark, catalog.Sources.MOVIELENS)

    for p in pipelines_to_run:
        if p == globals.Pipelines.DATA_ENGINEERING:
            train, serve = pipelines.data_engineering()
        if p == globals.Pipelines.DATA_SCIENCE:
            pipelines.data_science()


if __name__ == "__main__":
    # set time format for logger
    log_ts = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_ts)

    # load environment variables
    load_dotenv("../.env")

    # run `main`
    main()
