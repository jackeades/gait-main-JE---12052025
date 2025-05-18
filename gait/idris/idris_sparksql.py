import os
from typing import Optional, List

import pandas as pd

from .idris_base import IdrisRDB, IdrisException


class IdrisSparkSQL(IdrisRDB):
    def __init__(
            self,
            spark_session=None,
    ) -> None:
        """Initialize an Idris instance for Spark SQL.

        :param spark_session: Optional SparkSession to use.
        """
        try:
            from pyspark.sql import SparkSession
        except ImportError:
            raise IdrisException("Please run `pip install pyspark==3.5.5` to use IdrisSparkSQL.")
        self.spark = spark_session or SparkSession.builder.getOrCreate()

    @property
    def dialect(self) -> str:
        """The SQL dialect of the database.

        :return: The SQL dialect.
        """
        return "Spark SQL"

    def _get_create_table_columns(self, table_name: str) -> List[str]:
        return [f"{row.col_name} {row.data_type}" for row in self.spark.sql(f"DESCRIBE {table_name}").collect()]

    def execute_sql(self, sql: str) -> pd.DataFrame:
        """Execute SQL.

        :param sql: The SQL to execute.
        :return: The result of the SQL as a Pandas DataFrame.
        """
        return self.spark.sql(sql).toPandas()


class IdrisDatabricks(IdrisRDB):
    def __init__(
            self,
            server_hostname: Optional[str] = None,
            http_path: Optional[str] = None,
            access_token: Optional[str] = None,
            **kwargs,
    ) -> None:
        """Initialize an Idris instance using Databricks SQL Connector for Python.
        https://learn.microsoft.com/en-us/azure/databricks/dev-tools/python-sql-connector
        https://docs.databricks.com/en/dev-tools/databricks-connect/python/index.html

        :param server_hostname: The hostname of the Databricks server, or get env DATABRICKS_SERVER_HOSTNAME.
        :param http_path: The HTTP path of the Databricks server, or get env DATABRICKS_HTTP_PATH.
        :param access_token: The access token for the Databricks server, or get env DATABRICKS_TOKEN.
        :param kwargs: Additional arguments for the connection.
        """
        try:
            from databricks import sql
        except ImportError:
            raise IdrisException("Please run `pip install databricks` to use IdrisDatabricksConnect.")

        server_hostname = server_hostname or os.getenv("DATABRICKS_SERVER_HOSTNAME")
        if not server_hostname:
            raise IdrisException("Please provide a Databricks server hostname")

        http_path = http_path or os.getenv("DATABRICKS_HTTP_PATH")
        if not http_path:
            raise IdrisException("Please provide a Databricks HTTP path")

        access_token = access_token or os.getenv("DATABRICKS_TOKEN")
        if not access_token:
            raise IdrisException("Please provide a Databricks access token")

        self.connection = sql.connect(
            server_hostname=server_hostname,
            http_path=http_path,
            access_token=access_token,
            **kwargs,
        )

    def close(self) -> None:
        """Close the connection.
        """
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    @property
    def dialect(self) -> str:
        """The SQL dialect of the database.

        :return: The SQL dialect.
        """
        return "Spark SQL"

    def _get_create_table_columns(self, table_name: str) -> List[str]:
        with self.connection.cursor() as cursor:
            cursor.execute(f"DESCRIBE {table_name}")
            return [f"{row.col_name} {row.data_type}" for row in cursor.fetchall()]

    def execute_sql(self, sql: str) -> pd.DataFrame:
        """Execute SQL.

        :param sql: The SQL to execute.
        :return: The result of the SQL as a Pandas DataFrame.
        """
        with self.connection.cursor() as cursor:
            cursor.execute(sql)
            arrow_table = cursor.fetchall_arrow()
            return arrow_table.to_pandas()
