from abc import abstractmethod
from typing import Any, List

from .idris_abc import IdrisABC


class IdrisLLM(IdrisABC):
    """Abstract class to implement an LLM interface.
    """

    @abstractmethod
    def create_message(self, role: str, content: str) -> Any:
        """Create a message.

        :param role: The role of the message.
        :param content: The content of the message.
        """

    @abstractmethod
    def generate_sql(
            self,
            messages: List[Any],
            **kwargs
    ) -> str:
        """Generate SQL from a prompt.

        :param messages: A list of messages.
        :return: The generated SQL.
        """
