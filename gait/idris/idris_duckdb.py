import os
from typing import List

import pandas as pd

from .idris_base import IdrisRDB, IdrisException


class IdrisDuckDBException(IdrisException):
    def __init__(self) -> None:
        super().__init__("""Please run `pip install duckdb` to use IdrisDuckDB.""")


class IdrisDuckDB(IdrisRDB):
    def __init__(
            self,
            database: str,
            read_only: bool = True,
            config: dict = None,
            init_script: str = None,
    ) -> None:
        """Initialize an Idris DuckDB instance.

        :param database: The path to the DuckDB database. Required.
        :param read_only: Whether the database is read-only. Default is True.
        :param config: The DuckDB configuration dictionary. Optional.
        :param init_script: Optional initialization script to run.
        """
        try:
            import duckdb
        except ImportError:
            raise IdrisDuckDBException()

        self._duckdb = duckdb
        self.conn = self._duckdb.connect(
            database=os.path.expanduser(database),
            read_only=read_only,
            config=config or {},
        )
        self.conn.execute("INSTALL spatial;LOAD spatial;" + (init_script or ""))

    @property
    def dialect(self) -> str:
        """The SQL dialect of the database.

        :return: The SQL dialect.
        """
        return "DuckDB"

    def close(self) -> None:
        """Close open resources.
        """
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def _get_create_table_columns(self, table_name: str) -> List[str]:
        cols = [
            f"{row[0]} {row[1]}"
            for row in self.conn.sql(f"DESCRIBE {table_name}").fetchall()
        ]
        return cols

    def execute_sql(self, sql: str) -> pd.DataFrame:
        """Execute SQL.

        :param sql: The SQL to execute.
        :return: The result of the SQL as a Pandas DataFrame.
        """
        return self.conn.execute(sql.replace('"', "'")).df()
