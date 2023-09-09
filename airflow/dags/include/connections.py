import logging

from airflow import settings
from airflow.models import Connection


def create(args: dict):
    """ Creates an Airflow Connection"""

    conn = Connection(**args)
    session = settings.Session()
    conn_name = (
        session.query(Connection).filter(Connection.conn_id == conn.conn_id).first()
    )

    if str(conn_name) == str(conn.conn_id):
        logging.warning(f"Connection {conn.conn_id} already exists")
        return None

    session.add(conn)
    session.commit()
    logging.info(Connection.log_info(conn))
    logging.info(f"Connection {conn.conn_id} is created")
    session.close()
    return conn
