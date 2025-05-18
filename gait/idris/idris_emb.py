import logging
from abc import abstractmethod
from typing import List, Tuple, Any

import numpy as np

from .idris_abc import IdrisABC


class IdrisEmb(IdrisABC):
    """Abstract class to implement a knowledge base embedding interface.
    """

    @abstractmethod
    def dump_create_table(self) -> List[str]:
        """Dump the "CREATE TABLE" statements.

        :return: The "CREATE TABLE" statements.
        """

    @abstractmethod
    def dump_context(self) -> List[str]:
        """Dump the context.

        :return: The context.
        """

    @abstractmethod
    def dump_question_sql(self) -> List[Tuple[str, str]]:
        """Dump the question and SQL.

        :return: The question and SQL.
        """

    @abstractmethod
    def add_create_table(self, create_table: str) -> None:
        """Add a "CREATE TABLE" statement to an Idris instance.

        :param create_table: The "CREATE TABLE" statement.
        """

    @abstractmethod
    def add_context(self, context: str) -> None:
        """Add context to an Idris instance.

        :param context: The context to add.
        """

    @abstractmethod
    def add_question_sql(self, question: str, sql: str) -> None:
        """Add a question and its corresponding SQL to an Idris instance.

        :param question: The question.
        :param sql: The SQL.
        """

    @abstractmethod
    def load_context(self, context: List[str]) -> None:
        """Load context from a list.

        :param context: The file containing the context.
        """

    @abstractmethod
    def load_question_sql(self, question_sql: List[Tuple[str, str]]) -> None:
        """Load question and SQL from a list.

        :param question_sql: The file containing the question and SQL.
        """

    @abstractmethod
    def get_similar_create_table(
            self,
            prompt: str
    ) -> List[str]:
        """Get similar "CREATE TABLE" statements.

        :param prompt: The prompt.
        :return: list of max_embeddings similar create table statements.
        """

    @abstractmethod
    def get_similar_context(
            self,
            prompt: str
    ) -> List[str]:
        """Get similar context.

        :param prompt: The prompt.
        :return: list of max_embeddings similar contexts.
        """

    @abstractmethod
    def get_similar_question_sql(
            self,
            prompt: str
    ) -> List[Tuple[str, str]]:
        """Get similar question and SQL.

        :param prompt: The prompt.
        :return: list of max_embeddings similar question and SQL.
        """


class IdrisEmbInMemory(IdrisEmb):
    def __init__(
            self,
            max_context: int = 5,
            max_question_sql: int = 5,
    ) -> None:
        """Initialise an in-memory knowledge base embedding.

        :param max_context: The maximum number of context embeddings to store. Default: 5
        :param max_question_sql: The maximum number of question SQL embeddings to store. Default: 5
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.create_table: List[str] = []
        self.context: List[str] = []
        self.question_sql: List[Tuple[str, str]] = []
        self.context_embeddings: List[Any] = []
        self.question_sql_embeddings: List[Any] = []
        self.max_context = max_context
        self.max_question_sql = max_question_sql

    def _get_embedding(self, prompt: str) -> Any:
        """Get the embedding of a prompt. Abstract function to be implemented by the child class.
        """
        pass

    def _get_embeddings(self, prompt: List[str]) -> List[Any]:
        """Get the embeddings of a list of prompts. Abstract function to be implemented by the child class.
        """
        pass

    def add_create_table(self, create_table: str) -> None:
        self.create_table.append(create_table)

    def add_context(self, context: str) -> None:
        """Add context to an Idris instance.

        :param context: The context to add.
        """
        self.context.append(context)
        self.context_embeddings.append(self._get_embedding(context))

    def add_question_sql(self, question: str, sql: str) -> None:
        """Add a question and its corresponding SQL to an Idris instance.

        :param question: The question.
        :param sql: The SQL.
        """
        self.question_sql.append((question, sql))
        self.question_sql_embeddings.append(self._get_embedding(question))

    def load_context(self, context: List[str]) -> None:
        """Load context from a list.

        :param context: The file containing the context.
        """
        self.context.extend(context)
        self.context_embeddings.extend(self._get_embeddings(context))

    def load_question_sql(self, question_sql: List[Tuple[str, str]]) -> None:
        """Load question and SQL from a list.

        :param question_sql: The file containing the question and SQL.
        """
        self.question_sql.extend(question_sql)
        questions = [q for q, _ in self.question_sql]
        self.question_sql_embeddings.extend(self._get_embeddings(questions))

    def _similar_indices(
            self,
            prompt,
            embeddings
    ):
        scores = np.dot(embeddings, self._get_embedding(prompt))
        return np.argsort(scores)[::-1]

    def get_similar_create_table(
            self,
            prompt: str
    ) -> List[str]:
        return self.create_table

    def get_similar_context(
            self,
            prompt: str
    ) -> List[str]:
        indices = self._similar_indices(prompt, self.context_embeddings)
        return [self.context[i] for i in indices[:self.max_context]]  # [:-1]

    def get_similar_question_sql(
            self,
            prompt: str
    ) -> List[Tuple[str, str]]:
        indices = self._similar_indices(prompt, self.question_sql_embeddings)
        return [self.question_sql[i] for i in indices[:self.max_question_sql]]  # [:-1]

    def dump_create_table(self) -> List[str]:
        return self.create_table

    def dump_context(self) -> List[str]:
        return self.context

    def dump_question_sql(self) -> List[Tuple[str, str]]:
        return self.question_sql
