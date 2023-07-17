"""
Training Parameters
"""

# Raw train/prod data percentage
RAW: float = 0.05  # 0.10

# Training data percentage
TRAIN: float = 0.60  # 0.80

SEED: int = 42

from conf import paths


class ALS:
    """
    ALS Hyper-Parameters
    """

    RANK: list[int] = [8, 10, 12, 15, 18, 20]
    REG_INTERVAL: list[float] = [0.001, 0.2]
    COLD_START_STRATEGY: list[str] = ["drop"]
    MAX_ITER: list[int] = [5]


class Spark:
    spark_config: dict = {
        "spark.master": "spark://spark:7077",
        "spark.submit.deployMode": "client",
        "spark.driver.bindAddress": "0.0.0.0",
        "spark.driver.maxResultSize": "12g",
        "spark.driver.memory": "12g",
        "spark.executor.memory": "12g",
        "packages": "org.apache.hadoop:hadoop-aws:3.3.4",
        "spark.jars": paths.get_path(paths.ASSETS, "hadoop-aws-3.3.4.jar", as_string=True)
        + ","
        + paths.get_path(paths.ASSETS, "aws-java-sdk-bundle-1.12.506.jar", as_string=True)
        + ","
        + paths.get_path(paths.ASSETS, "hadoop-common-3.3.4.jar", as_string=True),
        "spark.hadoop.fs.s3a.path.style.access": "true",
        "spark.fs.s3a.path.style.access": "true",
        "spark.hadoop.fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",  # mlflow doesn't support s3a
        "spark.hadoop.fs.s3.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",
        "spark.hadoop.fs.s3a.aws.credentials.provider": "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
        "spark.fs.s3a.connection.ssl.enabled": "false",
        "spark.hadoop.fs.s3a.endpoint": "http://minio:9000",
        # "spark.hadoop.fs.s3a.access.key": "teste",
        # "spark.hadoop.fs.s3a.secret.key": "teste123",
    }


class Airflow:
    conn_config: dict = {
        "conn_id": "spark_connection",
        "conn_type": "spark",
        "host": "192.168.1.1",
        "login": "airflow",
        "password": "airflow",
        "port": 7077,
        "desc": "Spark Connection",
    }
    de_operator: dict = {
        "task_id": "de_spark",
        "conn_id": conn_config["conn_id"],
        "application": "include/collab_data_engineering.py",
        "executor_memory": "10G",
        "driver_memory": "10G",
        "executor_cores": 4,
    }
    ds_operator: dict = {
        "task_id": "ds_spark",
        "conn_id": conn_config["conn_id"],
        "application": "include/collab_data_science.py",
        "executor_memory": "10G",
        "driver_memory": "10G",
        "executor_cores": 4,
    }
