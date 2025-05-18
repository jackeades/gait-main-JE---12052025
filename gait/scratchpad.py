from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, Optional, Dict, Tuple

__SCRATCHPAD__ = "scratchpad"


class Scratchpad(ABC):
    """Abstract class to implement a scratchpad.

    A scratchpad is a key-value store that can be used to store intermediate results between function calls or agents.
    """

    def __enter__(self):
        """Enter a context manager.

        :return: An Idris instance.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit a context manager.

        Close open resources.
        """
        self.close()

    def __del__(self):
        """Delete an Idris instance.

        Close open resources.
        """
        self.close()

    @abstractmethod
    def __dict__(self) -> Dict[str, Any]:
        """Get all key-value pairs from the scratchpad as a dictionary.

        :return: The key-value pairs.
        """

    @abstractmethod
    def __iter__(self) -> Iterable[Tuple[str, Any]]:
        """Get all key-value pairs from the scratchpad as a iterator.

        :return: The key-value pairs.
        """

    @abstractmethod
    def __setitem__(self, key: str, value: Any) -> None:
        """Put a key-value pair into the scratchpad.

        :param key: The key.
        :param value: The value.
        """

    @abstractmethod
    def __getitem__(
            self,
            key: str
    ) -> Optional[Any]:
        """Get a value from the scratchpad.

        :param key: The key.
        :return: The value, or None if the key is not found.
        """

    @abstractmethod
    def get(
            self,
            key: str,
            default: Optional[Any] = None
    ) -> Optional[Any]:
        """Get a value from the scratchpad.

        :param key: The key.
        :param default: The default value to return if the key is not found.
        :return: The value, or the default if the key is not found.
        """

    def close(self) -> None:
        """Close open resources.
        """
        pass

    def reset(self) -> None:
        """Reset the scratchpad to its original state.
        """
        pass

    def load(self) -> None:
        """Load a scratchpad from a resource defined by the implementation.
        """
        pass

    def dump(self) -> None:
        """Dump a scratchpad to a resource defined by the implementation.
        """
        pass

    @classmethod
    def instance(cls) -> "Scratchpad":
        """Create an instance of Scratchpad.

        :return: An instance of ScratchpadInMemory.
        """
        return ScratchpadInMemory()


class ScratchpadInMemory(Scratchpad):
    def __init__(self, data: Optional[Dict[str, Any]] = None):
        """Initialize an in-memory scratchpad.

        :param data: Optional initial data for the scratchpad.
        """
        self._scratchpad = data or {}

    def __dict__(self) -> Dict[str, Any]:
        """Get all key-value pairs from the scratchpad.

        :return: The key-value pairs.
        """
        return self._scratchpad

    def __iter__(self) -> Iterable[Tuple[str, Any]]:
        """Get all key-value pairs from the scratchpad.

        :return: The key-value pairs.
        """
        return iter(self._scratchpad.items())

    def __setitem__(self, key: str, value: Any) -> None:
        """Put a key-value pair into the scratchpad.

        :param key: The key.
        :param value: The value.
        """
        self._scratchpad[key] = value

    def __getitem__(self, key: str) -> Optional[Any]:
        """Get a value from the scratchpad.

        :param key: The key.
        :return: The value, or None if the key is not found.
        """
        return self._scratchpad.get(key)

    def get(self, key: str, default: Optional[Any] = None) -> Optional[Any]:
        """Get a value from the scratchpad.

        :param key: The key.
        :param default: The default value to return if the key is not found.
        :return: The value, or the default if the key is not found.
        """
        return self._scratchpad.get(key, default)

    def close(self) -> None:
        """Close open resources.
        """
        self._scratchpad.clear()

    def reset(self) -> None:
        """Reset the scratchpad to its original state.
        """
        self._scratchpad.clear()
