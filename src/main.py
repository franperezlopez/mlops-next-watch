import logging
import logging.config
import os

import click
import mlflow
from dotenv import load_dotenv

from collaborative.pipelines import (
    data_engineering,
    data_science,
    data_validation,
    inference,
    monitoring,
)
from conf import globals, paths


@click.command()
@click.option(
    "-p",
    "--pipelines",
    "pipelines_to_run",
    default=[globals.Pipelines.DATA_SCIENCE],
    type=str,
    multiple=True,
    required=True,
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
@click.option(
    "-u",
    "--userids",
    "user_ids",
    default=[],
    type=int,
    multiple=True,
    nargs=1,
    help="Set User ID for inference.",
)
@click.option(
    "-n",
    "--nrecommendations",
    "n_recommendations",
    default=-1,
    type=int,
    nargs=1,
    help="Set a number of movie recommendations.",
)
def main(
    pipelines_to_run: str,
    experiment_name: str,
    user_ids: list[int],
    n_recommendations: int,
):
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

    for p in pipelines_to_run:
        if p == globals.Pipelines.DATA_ENGINEERING:
            data_engineering.run()
        if p == globals.Pipelines.DATA_SCIENCE:
            data_science.run()
        if p == globals.Pipelines.INFERENCE:
            inference.run(list(user_ids), n_recommendations)
        if p == globals.Pipelines.DATA_VALIDATION:
            data_validation.run()
        if p == globals.Pipelines.MONITOR:
            monitoring.run()


if __name__ == "__main__":
    # set time format for logger
    path_logging = paths.get_path(globals.Logs.CONFIG_FILE)
    logging.config.fileConfig(path_logging)

    # load environment variables
    load_dotenv(
        paths.get_path(
            paths.ENV,
            storage=globals.Storage.HOST,
        )
    )

    # run `main` test CI 2
    main()
