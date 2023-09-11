from datetime import datetime, timedelta

import pytz
from dotenv import load_dotenv
from include import connections

from airflow.models import DAG, Variable
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
connections.create(params.Airflow.spark_conn_config)


def create_first_train_variable():
    Variable.set("first_train_bool", "true")


default_args = {"owner": "nextwatch", "retries": 2, "retry_delay": timedelta(minutes=2)}
portugal_timezone = pytz.timezone("Europe/Lisbon")

with DAG(
    dag_id="collab_training_dag",
    description="Dag for DE and DS collab",
    start_date=datetime(2023, 1, 1),
    schedule_interval=timedelta(minutes=15),
    catchup=False,
    default_args=default_args,
) as training_dag:
    de = SparkSubmitOperator(**params.Airflow.de_operator, conf=params.Spark.spark_config)
    dv = SparkSubmitOperator(**params.Airflow.dv_operator, conf=params.Spark.spark_config)
    ds = SparkSubmitOperator(**params.Airflow.ds_operator, conf=params.Spark.spark_config)
    register_first_train_var = PythonOperator(
        task_id="register_first_train",
        python_callable=create_first_train_variable,
    )
    dv >> de >> ds >> register_first_train_var
