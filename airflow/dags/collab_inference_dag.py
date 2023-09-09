from datetime import datetime, timedelta

import pytz
from dotenv import load_dotenv
from include import connections
from include.collab_monitoring import run_monitoring
from include.first_train_sensor import FirstTrainSensor

from airflow.models import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.apache.spark.operators.spark_submit import (
    SparkSubmitOperator,
)
from conf import params, paths

load_dotenv(
    paths.get_path(
        paths.ENV,
    )
)

connections.create(params.Airflow.postgres_conn_config)
connections.create(params.Airflow.spark_conn_config)
connections.create(params.Airflow.aws_conn_config)

default_args = {"owner": "nextwatch", "retries": 2, "retry_delay": timedelta(minutes=2)}
portugal_timezone = pytz.timezone("Europe/Lisbon")

with DAG(
    dag_id="collab_inference_dag",
    description="Dag for batch Inference and Monitoring collab",
    start_date=datetime(2023, 1, 1),
    schedule_interval=timedelta(minutes=5),
    catchup=False,
    default_args=default_args,
) as inference_dag:

    # Checks if first train already happened!
    check_first_train = FirstTrainSensor(
        task_id="check_first_train",
        poke_interval=5,  # in seconds...
        dag=inference_dag,
    )
    inference = SparkSubmitOperator(
        **params.Airflow.inference_operator, conf=params.Spark.spark_config
    )
    monitoring = PythonOperator(
        task_id="monitoring_pandas", python_callable=run_monitoring
    )

    check_first_train >> inference >> monitoring
