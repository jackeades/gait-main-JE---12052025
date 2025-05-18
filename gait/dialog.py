from abc import ABC, abstractmethod
from typing import List, Iterator

from .types import MessageType
from .utils import u_message


class Dialog(ABC):
    """Abstract class to implement a sequence of messages.

    A message is a dictionary with the following keys:
        role:String
        content:String
    """

    def __enter__(self):
        """Enter a context manager.

        :return: A dialog instance.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit a context manager.

        Close open resources.
        """
        self.close()

    def __del__(self):
        """Delete a storage instance.

        Close open resources.
        """
        self.close()

    @abstractmethod
    def __iadd__(self, message: MessageType | str) -> "Dialog":
        """Add a message using the += operator.

        :param message: The chat message.
        :return: The current instance after adding the message.
        """

    @abstractmethod
    def __add__(self, message: MessageType | str) -> "Dialog":
        """Add a message using the + operator.

        :param message: The chat message.
        :return: A new instance of the dialog with the added message.
        """

    @abstractmethod
    def __iter__(self) -> Iterator[MessageType]:
        """Allow iteration over chat messages.

        :return: An iterator over the messages.
        """

    @abstractmethod
    def clone(self) -> "Dialog":
        """Create a clone of the dialog.

        :return: A new instance of the dialog.
        """

    @abstractmethod
    def close(self) -> None:
        """Close the storage.

        Close open resources.
        """

    @classmethod
    def instance(cls) -> "Dialog":
        """Create an instance of Dialog.

        :return: An instance of DialogInMemory.
        """
        return DialogInMemory()


class DialogInMemory(Dialog):
    def __init__(self, messages: List[MessageType] = None) -> None:
        """Initialize an in-memory dialog.

        :param messages: Initial messages.
        """
        self._messages = messages or []

    def __iadd__(self, message: MessageType | str) -> Dialog:
        """Save a message using the += operator.

        :param message: The message.
        :return: The current instance after adding the message.
        """
        match message:
            case str():
                self._messages.append(u_message(message))
            case _:
                self._messages.append(message)
        return self

    def __add__(self, message: MessageType | str) -> Dialog:
        """Save a message using the + operator.

        :param message: The message.
        :return: A new instance of the dialog with the added message.
        """
        dialog = self.clone()
        dialog += message
        return dialog

    def __iter__(self) -> Iterator[MessageType]:
        """Allow iteration over messages.

        :return: An iterator over the messages.
        """
        return iter(self._messages)

    def clone(self) -> "Dialog":
        """Create a clone of the dialog.

        :return: A new instance of the dialog.
        """
        return DialogInMemory(self._messages.copy())

    def close(self) -> None:
        """Close the storage.
        """
        return
