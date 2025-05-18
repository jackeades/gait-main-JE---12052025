from abc import ABCMeta, abstractmethod
from typing import List

from .fel_base import FELLine


class FELVSS(metaclass=ABCMeta):

    @abstractmethod
    def load(self, filename: str) -> None:
        pass

    @abstractmethod
    def dump(self, filename: str) -> None:
        pass

    @abstractmethod
    def create_fel0(self, fel_lines: List[FELLine]):
        pass

    @abstractmethod
    def create_fel1(self, fel_lines: List[FELLine]):
        pass

    @abstractmethod
    def create_fel2(self, fel_lines: List[FELLine]):
        pass

    @abstractmethod
    def query_fel0(self, query: str, n_results: int) -> List[FELLine]:
        pass

    @abstractmethod
    def query_fel1(self, query: str, n_results: int) -> List[FELLine]:
        pass

    @abstractmethod
    def query_fel2(self, query: str, n_results: int) -> List[FELLine]:
        pass
