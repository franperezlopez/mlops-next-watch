from dotenv import load_dotenv
from pyspark.sql import SparkSession

from collaborative.nodes import data_science_nodes, pre_processing_nodes
from conf import params, paths

load_dotenv(paths.get_path(paths.DOCKER_ENV))

spark_config: dict = {
    "spark.master": "spark://spark:7077",
    "spark.submit.deployMode": "client",
    "spark.driver.bindAddress": "0.0.0.0",
    "spark.driver.maxResultSize": "12g",
    "spark.driver.memory": "12g",
    "spark.executor.memory": "12g",
    "spark.jars": paths.get_path(paths.ASSETS, "hadoop-aws-3.3.4.jar", as_string=True)
    + ","
    + paths.get_path(paths.ASSETS, "aws-java-sdk-bundle-1.12.506.jar", as_string=True)
    + ","
    + paths.get_path(paths.ASSETS, "hadoop-common-3.3.4.jar", as_string=True),
    "spark.hadoop.fs.s3a.path.style.access": "true",
    "spark.hadoop.fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",  # mlflow doesn't support s3a
    "spark.hadoop.fs.s3.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",
    "spark.fs.s3a.connection.ssl.enabled": "false",
    "spark.hadoop.fs.s3a.endpoint": "http://minio:9000",
}

session = (
    SparkSession.builder.appName("collab-teste")
    .config(map=spark_config)
    .getOrCreate()
)

session.sparkContext.setLogLevel("INFO")

print("beofre df")
df = session.createDataFrame(["10", "11", "13"], "string").toDF("age")
print("after df")
df.write.mode("overwrite").parquet("s3a://data/teste.parquet")
print("after writing df")
session.stop()
