from typing import Optional, List

import pandas as pd

from .idris_base import IdrisRDB


class NotAllowedError(Exception):
    pass


class IdrisNoRDB(IdrisRDB):
    def __init__(
            self,
            dialect: str = None,
    ) -> None:
        """Initialize an Idris No Database instance.

        :param dialect: The SQL dialect of the database. Default = None.
        """
        self._dialect = dialect

    @property
    def dialect(self) -> str:
        """The SQL dialect of the database.

        :return: The SQL dialect.
        """
        return "generic" if self._dialect is None else self._dialect

    def _get_create_table_columns(
            self,
            table_name: str
    ) -> List[str]:
        return []

    def get_create_table(
            self,
            table_name: str,
            sep: str = ",",
            prefix: str = "\n",
            suffix: str = "\n",
    ) -> Optional[str]:
        """Describe a table.

        :param table_name: The name of the table.
        :param sep: The separator to use. Default is ','.
        :param prefix: The prefix to use before the column list after (. Default is '\n'.
        :param suffix: The suffix to use after the columns list before ). Default is '\n'.
        :return: The "CREATE TABLE" statement of the table.
        """
        return None

    def execute_sql(self, sql: str) -> pd.DataFrame:
        """Execute SQL.

        :param sql: The SQL to execute.
        :return: The result of the SQL as a Pandas DataFrame.
        """
        raise NotAllowedError("Not allowed to execute SQL.")
