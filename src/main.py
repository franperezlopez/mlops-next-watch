from pathlib import Path
import click

from loguru import logger

from src.pipelines.training import Train
from src.pipelines.monitoring import Monitor
from src.pipelines.inference import Predict
from src.settings import get_settings
from src.pipelines import Pipelines

@click.command()
@click.option("-p", "--pipeline", default=str(Pipelines.TRAINING), required=True,
              type=click.Choice([p.value for p in Pipelines]), 
              help="Set pipeline to run")
@click.option("-i", "--input_file", type=click.Path(exists=True), 
              help="Set input file (only valid for TRAINING and INFERENCE pipelines)")
def main(pipeline: Pipelines, input_file: str):
    """Main Program

    Args:
        pipeline (str): pipeline names to be run
        input_file (str): input file path
    """
    logger.info("Taxi Fare Prediction CLI")

    # mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    # mlflow.set_experiment(experiment_name)

    logger.info(f"Running {pipeline} pipeline")
    match Pipelines(pipeline):
        case Pipelines.INFERENCE:
            Predict(get_settings()).run(Path(input_file))
        case Pipelines.MONITOR:
            Monitor(get_settings()).run()
        case Pipelines.TRAINING:
            Train(get_settings()).run(Path(input_file))
        case _:
            logger.error("Invalid pipeline name")
            raise ValueError("Invalid pipeline name")


if __name__ == "__main__":
    main()
