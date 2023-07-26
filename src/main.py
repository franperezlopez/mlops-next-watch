import logging
import logging.config
import os

import click
import mlflow
from dotenv import load_dotenv

from collaborative.pipelines import pipelines
from conf import globals, paths


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
    logging.getLogger("mlflow").setLevel(logging.DEBUG)
    logging.getLogger("pyspark").setLevel(logging.DEBUG)
    logger.info("Running Next-Watch")


    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(experiment_name)

    mlflow.spark.autolog(disable=True)

    print(pipelines_to_run)
    for p in pipelines_to_run:
        print(p)
        if p == globals.Pipelines.DATA_ENGINEERING:
            pipelines.data_engineering()
        if p == globals.Pipelines.DATA_SCIENCE:
            pipelines.data_science()


if __name__ == "__main__":
    # set time format for logger
    path_logging = paths.get_path(globals.Logs.CONFIG_FILE)
    print(path_logging)
    logging.config.fileConfig(path_logging)

    # load environment variables

    print(type(os.environ["DOCKER_RUNNING"]))
    # print(
    #    paths.DOCKER_ENV
    #    if "DOCKER_RUNNING" in os.environ and os.environ["DOCKER_RUNNING"] == "true"
    #    else paths.ENV
    # )
    load_dotenv(
        paths.get_path(
            paths.ENV,
            storage=globals.Storage.HOST,
        )
    )

    # run `main`
    main()
