"""
Training Parameters
"""

# Raw train/prod data percentage
RAW: float = 0.05

# Training data percentage
TRAIN: float = 0.80

SEED: int = 42

import os

from dotenv import load_dotenv

from conf import paths, globals

load_dotenv(
    paths.get_path(
        paths.ENV,
    )
)


class ALS:
    """
    ALS Hyper-Parameters
    """

    RANK: list[int] = [8, 10, 12, 15, 18, 20]
    REG_INTERVAL: list[float] = [0.001, 0.2]
    COLD_START_STRATEGY: list[str] = ["drop"]
    MAX_ITER: list[int] = [5]


class Monitoring:
    TARGET_DRIFT_THR = 0.30
    DATA_DRIFT_THR = 0.10


class Spark:
    spark_config: dict = {
        "spark.master": "spark://spark:7077",
        "spark.submit.deployMode": "client",
        "spark.local.dir": paths.get_path(
            paths.SPARKML_TMP_DIR, storage=globals.Storage.DOCKER, as_string=True
        ),
        "spark.modify.acls": "*",  # * -> all users (we want airflow to have write permissions)
        "spark.driver.bindAddress": "0.0.0.0",
        "spark.driver.maxResultSize": "4g",
        "spark.driver.memory": "4g",
        "spark.executor.memory": "4g",
        "packages": "org.apache.hadoop:hadoop-aws:3.3.4",
        "spark.jars": paths.get_path(paths.ASSETS, "hadoop-aws-3.3.4.jar", as_string=True)
        + ","
        + paths.get_path(paths.ASSETS, "aws-java-sdk-bundle-1.12.506.jar", as_string=True)
        + ","
        + paths.get_path(paths.ASSETS, "hadoop-common-3.3.4.jar", as_string=True)
        + ","
        + paths.get_path(paths.ASSETS, "postgresql-42.6.0.jar", as_string=True),
        "spark.driver.extraClassPath": paths.get_path(
            paths.ASSETS, "postgresql-42.6.0.jar", as_string=True
        ),
        "spark.hadoop.fs.s3a.path.style.access": "true",
        "spark.fs.s3a.path.style.access": "true",
        "spark.hadoop.fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",
        "spark.hadoop.fs.s3.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",
        "spark.hadoop.fs.s3a.aws.credentials.provider": "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
        "spark.fs.s3a.connection.ssl.enabled": "false",
        "spark.hadoop.fs.s3a.endpoint": f"http://{os.getenv('MINIO_IP')}:{os.getenv('MINIO_PORT_SERVER')}",
    }


class Airflow:
    spark_conn_config: dict = {
        "conn_id": "spark_connection",
        "conn_type": "spark",
        "host": "spark",
        "port": 7077,
        "description": "Spark Connection",
    }
    postgres_conn_config: dict = {
        "conn_id": "postgres_connection",
        "conn_type": "postgres",
        "host": f"{os.getenv('postgres_ip')}",
        "login": f"{os.getenv('pguser')}",
        "password": f"{os.getenv('pgpassword')}",
        "port": f"{os.getenv('postgres_port')}",
        "schema": f"{os.getenv('postgres_app_database')}",
    }
    aws_conn_config: dict = {
        "conn_id": "aws_connection",
        "conn_type": "aws",
        "login": f"{os.getenv('MINIO_ACCESS_KEY')}",
        "password": f"{os.getenv('MINIO_SECRET_ACCESS_KEY')}",
    }
    de_operator: dict = {
        "task_id": "de_spark",
        "conn_id": spark_conn_config["conn_id"],
        "application": "airflow/dags/include/collab_data_engineering.py",
    }
    ds_operator: dict = {
        "task_id": "ds_spark",
        "conn_id": spark_conn_config["conn_id"],
        "application": "airflow/dags/include/collab_data_science.py",
    }
    dv_operator: dict = {
        "task_id": "dv_spark",
        "conn_id": spark_conn_config["conn_id"],
        "application": "airflow/dags/include/collab_data_validation.py",
    }
    inference_operator: dict = {
        "task_id": "inference_spark",
        "conn_id": spark_conn_config["conn_id"],
        "application": "airflow/dags/include/collab_inference.py",
    }
