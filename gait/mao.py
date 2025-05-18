from typing import Optional, Generator

from .agent import Agent
from .dialog import Dialog, DialogInMemory
from .observer import Observer, ObserverNoop
from .scratchpad import Scratchpad, ScratchpadInMemory
from .utils import u_message


class MAO:
    """Multi Agent Orchestrator.
    """

    def __init__(
            self,
            agent: Agent,
            dialog: Optional[Dialog] = None,
            scratchpad: Optional[Scratchpad] = None,
            observer: Observer = None,
    ) -> None:
        """Initialize the Multi Agent Orchestrator.

        :param agent: The initial agent.
        :param scratchpad: Optional Scratchpad instance. If not provided, an in-memory scratchpad is used.
        :param observer: Optional Observer instance. If not provided, a no-op observer is used.
        """
        if agent is None:
            raise ValueError("Agent is required.")
        self.agent = agent
        self.dialog = dialog or DialogInMemory()
        self.scratchpad = scratchpad or ScratchpadInMemory()
        self.observer = observer or ObserverNoop()
        self._terminate = False

    def terminate(self):
        """Terminate the MAO iteration.
        """
        self._terminate = True

    def __call__(
            self,
            prompt: str,
            iterations: int = 10,
    ) -> Generator:
        """Start the MAO iterations.

        :param prompt: The initial prompt.
        :param iterations: The maximum number of iterations. Defaults to 10.
        """
        self.dialog += u_message(prompt)
        self._terminate = False
        iteration = 0
        while not self._terminate and iteration < iterations:
            iteration += 1
            self.observer.on_iteration(iteration, self.agent.name)
            agent_response = self.agent(self.dialog, self.scratchpad, self.observer)
            self.agent = agent_response.agent
            yield agent_response
