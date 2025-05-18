from abc import ABC


class IdrisABC(ABC):
    """The base class for Idris.
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

    def close(self) -> None:
        """Close open resources.
        """
        pass
