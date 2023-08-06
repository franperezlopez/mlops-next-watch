import os
import pickle

import kafka
from dotenv import load_dotenv

from conf import globals, paths
from utils.psycopg_handler import PsycopgHandler

load_dotenv(
    paths.get_path(
        paths.ENV,
        storage=globals.Storage.DOCKER,
    )
)
table_name = os.getenv("POSTGRES_RECOMMENDATIONS_TABLE")
recommendations_topic = os.getenv("KAFKA_RECOMMENDATIONS_TOPIC")

psycopg = PsycopgHandler(
    os.getenv("POSTGRES_USER"),
    os.getenv("POSTGRES_PASSWORD"),
    os.getenv("POSTGRES_IP"),
    os.getenv("POSTGRES_PORT"),
    os.getenv("POSTGRES_APP_DATABASE"),
)

consumer = kafka.KafkaConsumer(
    bootstrap_servers=f"{os.getenv('KAFKA_IP')}:{os.getenv('KAFKA_PORT')}",
    group_id="inference-group",
    api_version=(3, 5, 1),
)

consumer.subscribe([recommendations_topic])

for message in consumer:
    if message.topic == recommendations_topic:
        recommendations = [
            (i+1, userId, movieId, prediction, rank)
            for i, (userId, movieId, prediction, rank) in enumerate(pickle.loads(message.value))
        ]
        print(recommendations)
        psycopg.update_db(
            f"""
                INSERT INTO {os.getenv("POSTGRES_RECOMMENDATIONS_TABLE")}
                (id, userid, movieid, prediction, rank)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (userid, rank) DO UPDATE
                SET id = EXCLUDED.id, movieid = EXCLUDED.movieid, prediction = EXCLUDED.prediction, rank = EXCLUDED.rank;
            """,
            recommendations,
        )
