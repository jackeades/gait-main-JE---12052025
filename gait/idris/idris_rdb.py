from abc import abstractmethod
from typing import Optional, List

import pandas as pd

from .idris_abc import IdrisABC


class IdrisRDB(IdrisABC):
    """Abstract class to implement a database interface.
    """

    @property
    @abstractmethod
    def dialect(self) -> str:
        """The SQL dialect of the database.

        :return: The SQL dialect.
        """

    @abstractmethod
    def _get_create_table_columns(
            self,
            table_name: str
    ) -> List[str]:
        """Get the columns of a table. OVERRIDE THIS.

        :param table_name: The name of the table.
        :return: The columns of the table.
        """

    def get_create_table(
            self,
            table_name: str,
            sep: str = ",",
            prefix: str = "\n",
            suffix: str = "\n",
    ) -> Optional[str]:
        """Return the "CREATE TABLE" statement of the table.

        :param table_name: The name of the table.
        :param sep: The separator to use. Default is ','.
        :param prefix: The prefix to use before the column list after (. Default is '\n'.
        :param suffix: The suffix to use after the columns list before ). Default is '\n'.
        :return: The "CREATE TABLE" statement of the table.
        """
        columns = sep.join(self._get_create_table_columns(table_name))
        return f"""CREATE TABLE {table_name} ({prefix}{columns}{suffix})"""

    @abstractmethod
    def execute_sql(self, sql: str) -> pd.DataFrame:
        """Execute SQL.

        :param sql: The SQL to execute.
        :return: The result of the SQL as a Pandas DataFrame.
        """
