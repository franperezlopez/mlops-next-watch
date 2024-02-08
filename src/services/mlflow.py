from pathlib import Path
from typing import Any, Callable
import mlflow
from loguru import logger
from functools import partial
from src.model.metrics import EVAL_METRIC, TRAIN_METRIC, CompareMetrics, CompareMetricsTrainEval
from src.model.settings import METRIC_IMPROVEMENT_THRESHOLD, METRIC, TARGET_COL
from src.services.utils import pd_read


def download_model(registered_model_name: str = "model") -> Any:
    """
    It downloads the model from the MLflow model registry and returns the model object

    :param registered_model_name: The name of the registered model in MLflow, defaults to model
    :type registered_model_name: str (optional)
    :param version: The version of the model to download. If you don't specify this parameter, the
    :return: The model object
    """

    runid = get_latest_run_id(registered_model_name)
    model = mlflow.sklearn.load_model(model_uri=f"runs:/{runid}/model")
    return model


def get_latest_run_id(model_name: str) -> str:
    """
    It returns the run ID of the latest version of the model with the given name

    :param model_name: The name of the model
    :type model_name: str
    :return: The run_id of the latest version of the model.
    """
    client = mlflow.tracking.MlflowClient()
    reg_models = client.search_registered_models(f"name = '{model_name}'")
    return list(map(lambda x: x.latest_versions[0].run_id, reg_models))[0]


def evaluate_run_id_model_on_data(run_id: str, model_run_id: str, filepath: Path, model_name: str):
    """
    The function loads the model from the model run ID, and then evaluates the model on the test data

    :param run_id: The run_id of the model to be evaluated. In case of training,
    the same run_id should be used; otherwise, parameter run_id should be none and a new run_id is generated
    :type run_id: str
    :param model_run_id: The run_id of the model you want to evaluate
    :type model_run_id: str
    :param filepath: The path to the data file
    :type filepath: Path
    """
    # run for evaluation
    # in case of training, the same run_id should be used
    # otherwise, parameter run_id should be none and a new run_id is generated
    with mlflow.start_run(run_id=run_id) as run:
        # Load the test data
        test_data = pd_read(filepath)

        model_uri = f"runs:/{model_run_id}/model"
        logger.info(f"model uri: {model_uri}")
        # Log test metrics
        mlflow.evaluate(
            model_uri,
            data=test_data,
            targets=TARGET_COL,
            model_type="regressor",  # "regressor" or "classifier"
            dataset_name=f"{model_name}_test",
            evaluator_config={"log_model_explainability": False},
        )
        
        evaluate_run_id = run.info.run_id
    return evaluate_run_id


def send_metric_using_policy(appins: AppInsightsLogger, model_name: str, run_id: str, 
                             func_policy: Callable, 
                             threshold_percentage: float = METRIC_IMPROVEMENT_THRESHOLD, **kwargs):
    """
    It takes a function that returns a `CompareMetrics` object and sends the metric to Application
    Insights if the metric is better than the previous
    
    :param appins: Logger
    :type appins: AppInsightsLogger
    :param model_name: The name of the model
    :type model_name: str
    :param run_id: The run_id of the run that you want to compare against
    :type run_id: str
    :param func_policy: This is the function that will be used to compare the metrics
    :type func_policy: Callable
    :param threshold_percentage: This is the threshold percentage of the difference between the training
    and reference metrics. If the difference is greater than this threshold, then the metric is sent to
    Application Insights
    :type threshold_percentage: float
    """
    metric_comparer: CompareMetrics = func_policy(model_name=model_name, run_id=run_id, **kwargs)
    if metric_comparer.is_worse_metric(EVAL_METRIC):
        eval_value = metric_comparer.data[EVAL_METRIC]
        train_value = metric_comparer.data[TRAIN_METRIC]
        # threshold logic can be moved to monitoring component
        if (abs(eval_value - train_value) / train_value) * 100.0 >= threshold_percentage:
            logger.info(f"Sendind customMetrics - {metric_comparer.metric} : {eval_value}")
            appins.configure_metric(name=metric_comparer.metric)
            appins.log_metric(value=eval_value)
    else:
        logger.info(f"NOT sending customMetrics")


def register_model_using_policy(model_name: str, run_id: str, func_policy: Any, **kwargs):
    """
    If the model registry policy is mandatory, then register the model using the run id and model name

    :param model_name: The name of the model to be registered
    :type model_name: str
    :param run_id: The run ID of the run that produced the model
    :type run_id: str
    :param deploy_flag: This is a boolean flag that indicates whether the model should be registered or
    not
    :type deploy_flag: bool
    :return: The model version
    """
    if func_policy(model_name=model_name, run_id=run_id, **kwargs):
        with mlflow.start_run(run_id=run_id):
            logger.info("Registering model", model_name)

            # register model using mlflow model
            model_uri = f"runs:/{run_id}/model"
            model_version = mlflow.register_model(model_uri, model_name)
            return model_version
    else:
        logger.warning("Model will not be registered!")


def _get_current_model_metric(run_id, metric, client):
    current_value = None
    try:
        logger.info(f"current run_id: {run_id}")
        current_run = client.get_run(run_id)
        logger.info(f"current metrics: {current_run.data.metrics}")
        current_value = current_run.data.metrics[metric]
    except Exception as ex:
        logger.warning(ex)
    return current_value


def _get_latest_model_metric(model_name, metric, client):
    try:
        latest_run_id = get_latest_run_id(model_name)
        logger.info(f"latest run_id: {latest_run_id}")
        latest_run = client.get_run(latest_run_id)
        logger.info(f"latest metrics: {latest_run.data.metrics}")
        latest_value = latest_run.data.metrics[metric]
    except Exception as ex:
        logger.warning(ex)
        latest_value = None
    return latest_value
