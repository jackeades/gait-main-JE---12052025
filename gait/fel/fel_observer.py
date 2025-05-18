from abc import ABC, abstractmethod

from .fel_base import FELLine, FEL0, FEL1, FEL2


class FELObserverABC(ABC):
    @abstractmethod
    def on_fel_line(self, fel_line: FELLine) -> None:
        pass

    @abstractmethod
    def on_fel0(self, fel0: FEL0) -> None:
        pass

    @abstractmethod
    def on_fel1(self, fel1: FEL1) -> None:
        pass

    @abstractmethod
    def on_fel2(self, fel2: FEL2) -> None:
        pass


class FELObserverNoop(FELObserverABC):
    def on_fel_line(self, fel_line: FELLine) -> None:
        pass

    def on_fel0(self, fel0: FEL0) -> None:
        pass

    def on_fel1(self, fel1: FEL1) -> None:
        pass

    def on_fel2(self, fel2: FEL2) -> None:
        pass


class FELObserverConsole(FELObserverABC):
    def on_fel_line(self, fel_line: FELLine) -> None:
        print("\n".join([fel_line.line, str(fel_line.fel)]))

    def on_fel0(self, fel0: FEL0) -> None:
        print(fel0)

    def on_fel1(self, fel1: FEL1) -> None:
        print(fel1)

    def on_fel2(self, fel2: FEL2) -> None:
        print(fel2)
