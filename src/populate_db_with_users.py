from pyspark.sql import SparkSession

from conf import catalog, globals, paths

session = (
    SparkSession.builder.appName("Populate DB with existing users")
    .config(
        "spark.jars",
        paths.get_path(
            paths.ASSETS,
            "postgresql-42.6.0.jar",
            as_string=True,
            storage=globals.Storage.HOST,
        ),
    )
    .config(
        "spark.driver.extraClassPath",
        paths.get_path(
            paths.ASSETS,
            "postgresql-42.6.0.jar",
            as_string=True,
            storage=globals.Storage.HOST,
        ),
    )
    .getOrCreate()
)

url = "jdbc:postgresql://localhost:5432/app"
properties = {
    "user": "admin",
    "password": "admin",
    "driver": "org.postgresql.Driver",
}
table_name = "users"

ext_filepath = paths.get_path(
    paths.DATA_01EXTERNAL,
    catalog.Sources.MOVIELENS,
    catalog.Datasets.RATINGS,
    suffix=catalog.FileFormat.CSV,
    as_string=True,
)

dataset = session.read.load(
    ext_filepath, format=catalog.FileFormat.CSV[1:], header=True, inferSchema=True
)
dataset[["userId"]].distinct().write.jdbc(url, table_name, mode="overwrite", properties=properties)
