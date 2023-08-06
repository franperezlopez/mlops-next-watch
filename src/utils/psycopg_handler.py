import psycopg2
from psycopg2.extras import execute_values


class PsycopgHandler:
    def __init__(self, user, password, host, port, database) -> None:
        self.connection = psycopg2.connect(
            user=user,
            password=password,
            host=host,
            port=port,
            database=database,
        )
        self.cursor = self.connection.cursor()

    def update_db(self, query: str, args: list):
        for arg in args:
            try:
                self.cursor.execute(query, arg)
                self.connection.commit()
            except Exception as error:
                print(error)
                print("Exception! Rolling back...")
                self.connection.rollback()

    def read_db(self, query: str):
        self.cursor.execute(query)
        return self.cursor.fetchall()

    def __exit__(self):
        self.cursor.close()
        self.connection.close()
