from typing import List

import pandas as pd

from .idris_base import IdrisRDB, IdrisException


class IdrisPostGIS(IdrisRDB):
    def __init__(
            self,
            database: str,
            username: str,
            password: str,
            host: str = "localhost",
            port: int = 5432,
            **kwargs
    ) -> None:
        """Initialize an Idris PostGIS instance.

        :param database: The path to the PostGIS database. Required.
        :param username: The username of the PostGIS database. Required.
        :param password: The password of the PostGIS database. Required.
        :param host: The host of the PostGIS database. Default = localhost.
        :param port: The port of the PostGIS database. Default = 5432.
        :param kwargs: Additional arguments for the connection.
        """

        try:
            import psycopg2
        except ImportError:
            raise IdrisException("Please run `pip install psycopg2-binary` to use IdrisPostGIS.")

        self.conn = psycopg2.connect(
            database=database,
            user=username,
            password=password,
            host=host,
            port=port,
            **kwargs,
            # cursor_factory=psycopg2.extras.DictCursor
        )
        self.conn.autocommit = True

    @property
    def dialect(self) -> str:
        """The SQL dialect of the database.

        :return: The SQL dialect.
        """
        return "PostGIS"

    def close(self) -> None:
        """Close open resources.
        """
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def _get_create_table_columns(self, table_name: str) -> List[str]:
        with self.conn.cursor() as cursor:
            cursor.execute(f"""WITH L as (SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = %s),
R as (SELECT f_geometry_column, srid, type
FROM geometry_columns
WHERE f_table_name = %s)
SELECT
column_name,
CASE WHEN srid is NULL
THEN UPPER(data_type)
ELSE CONCAT('GEOMETRY(',type,',',srid,')')
END as data_type
FROM l
LEFT JOIN r
ON l.column_name = r.f_geometry_column""".strip(), (table_name, table_name))
            return [f"{row[0]} {row[1]}" for row in cursor]

    def execute_sql(self, sql: str) -> pd.DataFrame:
        """Execute SQL.

        :param sql: The SQL to execute.
        :return: The result of the SQL as a Pandas DataFrame.
        """
        with self.conn.cursor() as cursor:
            cursor.execute(sql)
            results = cursor.fetchall()
            return pd.DataFrame(
                results,
                columns=[desc[0] for desc in cursor.description]
            )
