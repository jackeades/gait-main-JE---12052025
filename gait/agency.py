from typing import Generator

from .dialog import Dialog, DialogInMemory
from .observer import Observer, ObserverNoop
from .operator import Operator
from .scratchpad import Scratchpad, ScratchpadInMemory
from .utils import u_message


class Agency(object):
    def __init__(
            self,
            operator: Operator,
            dialog: Dialog = None,
            scratchpad: Scratchpad = None,
            observer: Observer = None,
    ) -> None:
        """Initialize the Agency.

        :param operator: The operator to use.
        :param dialog: Optional Dialog instance. If not provided, an in-memory dialog is used.
        :param scratchpad: Optional Scratchpad instance. If not provided, an in-memory scratchpad is used.
        :param observer: Optional Observer instance. If not provided, a no-op observer is used.
        """
        if operator is None:
            raise ValueError("Parameter operator is required.")
        self.operator = operator
        self.dialog = dialog or DialogInMemory()
        self.scratchpad = scratchpad or ScratchpadInMemory()
        self.listener = observer or ObserverNoop()
        self._terminate = False

    def terminate(self) -> None:
        """Terminate the agency iteration.
        """
        self._terminate = True

    def __call__(
            self,
            prompt: str,
            iterations: int = 10,
    ) -> Generator:
        """Start the agency iterations.

        :param prompt: The initial prompt.
        :param iterations: The maximum number of iterations. Defaults to 10.
        """
        self.dialog += u_message(prompt)
        self._terminate = False
        iteration = 0
        while not self._terminate and iteration < iterations:
            iteration += 1
            agent = self.operator.route(self.dialog)
            # TODO - Check if agent is None
            yield agent(self.dialog, self.scratchpad, self.listener)
