#
# Enable the implementation of a flow-based programming model
# https://www.anthropic.com/engineering/building-effective-agents
#
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Iterable, ClassVar, Literal

from ..scratchpad import Scratchpad


class Transition:
    def __init__(self, action: str, orig: "Node") -> None:
        """Initialize a transition object.

        This object is used to define the transition from one node to another.

        :param action: The action that triggers the transition.
        :param orig: The original node.
        """
        self.action = action
        self.orig = orig

    def __rshift__(self, dest: "Node") -> "Node":
        self.orig.set_action_node(self.action, dest)
        return dest


class Node(ABC):
    DEFAULT: ClassVar[str] = "default"

    def __init__(
            self,
    ) -> None:
        """Initialize a node object.
        """
        self._action_node_dict: Dict[str, "Node"] = {}

    def set_action_node(self, action: str, node: "Node") -> None:
        """Set the action and the node for the transition.

        :param action: The action that triggers the transition.
        :param node: The node to transition to.
        """
        self._action_node_dict[action] = node

    @abstractmethod
    def exec(
            self,
            scratchpad: Scratchpad,
    ) -> str:
        """Execute the node and return the action.

        :param scratchpad: A scratchpad instance.

        :return: The action that triggers the next node.
        """
        pass

    def __call__(
            self,
            scratchpad: Scratchpad,
    ) -> "Node":
        """Invoke the node and return the next node.

        :param scratchpad: A scratchpad instance.

        :return: The next node is based on the action returned by the `exec` method.
        """
        action = self.exec(scratchpad)
        return self._action_node_dict.get(action, None)

    def __rshift__(self, node: "Node") -> "Node":
        """Define the default transition from this node to another node.
        """
        self._action_node_dict[Node.DEFAULT] = node
        return node

    def __sub__(self, action: str) -> Transition:
        """Define a transition from this node to another node based on an action.

        :param action: The action that triggers the transition.
        :return: A transition object.
        """
        if not isinstance(action, str):
            raise TypeError("Action must be a string")
        return Transition(action, self)


class ANode(Node):
    """Create an abstract node class for asynchronous execution.
    """

    @abstractmethod
    async def exec(
            self,
            scratchpad: Scratchpad,
    ) -> str:
        """Execute the node and return the action.

        :param scratchpad: A scratchpad instance.
        """
        pass

    async def __call__(
            self,
            scratchpad: Scratchpad,
    ) -> Node:
        """Invoke the node and return the next node.

        :param scratchpad: A scratchpad instance.
        """
        action = await self.exec(scratchpad)
        return self._action_node_dict.get(action, None)


class PNode(ANode):
    """Create an abstract node class for parallel execution.
    """

    @abstractmethod
    async def _prep(
            self,
            scratchpad: Scratchpad
    ) -> Iterable:
        """Create iterable of items to execute in parallel.

        :param scratchpad: A scratchpad instance.
        """
        pass

    @abstractmethod
    async def _exec(
            self,
            scratchpad: Scratchpad,
            item: Any,
    ) -> Any:
        """Execute an item from the iterator and return the result.

        :param scratchpad: A scratchpad instance.
        :param item: An item from the iterator.
        """
        pass

    @abstractmethod
    async def _post(
            self,
            scratchpad: Scratchpad,
            exec_resp: Iterable
    ) -> str:
        """Process the results of the executed items and return the action.

        :param scratchpad: A scratchpad instance.
        :param exec_resp: Iterable of results from the executed items.
        """
        pass

    async def exec(
            self,
            scratchpad: Scratchpad,
    ) -> str:
        """Execute the node and return the action.
        """
        prep_resp = await self._prep(scratchpad)
        # https://docs.python.org/3/library/asyncio-task.html#running-tasks-concurrently
        exec_resp = await asyncio.gather(*(self._exec(scratchpad, _) for _ in prep_resp))
        return await self._post(scratchpad, exec_resp)


class FlowObserverABC(ABC):
    @abstractmethod
    def on_flow_start(self, flow: "FlowABC") -> None:
        """Called when flow starts.
        :param flow:
        """
        pass

    @abstractmethod
    def on_flow_end(self, flow: "FlowABC") -> None:
        """Called when the flow ends.
        :param flow:
        """
        pass

    @abstractmethod
    def on_node_start(self, node: Node) -> None:
        """Called when a node starts executing.

        :param node: The node that is starting.
        """
        pass

    @abstractmethod
    def on_node_end(self, node: Node) -> None:
        """Called when a node ends executing.

        :param node: The node that has finished.
        """
        pass


class FlowObserverNoop(FlowObserverABC):
    """No-op flow observer.
    """

    def on_flow_start(self, flow: "FlowABC") -> None:
        return

    def on_flow_end(self, flow: "FlowABC") -> None:
        return

    def on_node_start(self, node: Node) -> None:
        return

    def on_node_end(self, node: Node) -> None:
        return


class FlowObserverConsole(FlowObserverABC):
    """Flow observer that prints the flow events to the console.
    """

    def on_flow_start(self, flow: "FlowABC") -> None:
        print("Flow started...")

    def on_flow_end(self, flow: "FlowABC") -> None:
        print("Flow ended.")

    def on_node_start(self, node: Node) -> None:
        print(f"{node.__class__.__name__} started...")

    def on_node_end(self, node: Node) -> None:
        print(f"{node.__class__.__name__} ended.")


@dataclass
class FlowState:
    node: Node
    scratchpad: Scratchpad


class FlowABC(ABC):
    def __init__(
            self,
            start: Node,
            scratchpad: Scratchpad = None,
            observer: FlowObserverABC = None,
            **kwargs
    ) -> None:
        """Initialize a flow object.

        :param start: The starting node of the flow.
        :param scratchpad: The scratchpad object to use. If None, a new in-memory instance is created.
        """
        self.scratchpad = scratchpad or Scratchpad.instance()
        self.observer = observer or FlowObserverNoop()
        self.start = start
        self._prev = start
        self._curr = None
        self._put_kwargs(**kwargs)

    def __setitem__(self, key: str, value: Any) -> None:
        """Put a key-value pair into the scratchpad.

        :param key: The key.
        :param value: The value.
        """
        self.scratchpad[key] = value

    def __getitem__(
            self,
            key: str
    ) -> Any:
        """Get a value from the scratchpad.

        :param key: The key.
        :return: The value.
        """
        return self.scratchpad[key]

    def _put_kwargs(self, **kwargs) -> None:
        """Put key-value pairs into the scratchpad.
        """
        for key, value in kwargs.items():
            self.scratchpad[key] = value

    @abstractmethod
    def __call__(
            self,
            **kwargs,
    ) -> "FlowABC":
        """Invoke the flow.
        """
        pass

    def dump_state(self) -> FlowState:
        """Dump the current state of the flow.

        :return: The current state of the flow.
        """
        return FlowState(self._prev, self.scratchpad)

    def load_state(self, state: FlowState) -> None:
        """Load the state of the flow.

        :param state: The state to load.
        """
        self._curr = state.node
        self.scratchpad = state.scratchpad

    def display_markdown(
            self,
            direction: Literal["LR", "TD"] = "LR",
    ) -> None:
        """Display mermaid Markdown of the internal node structure.

        :param direction: The direction of the flowchart. The default is "LR" (left to right).
        """

        from IPython.display import Markdown, display

        def _node_to_str(node_: Node) -> str:
            if isinstance(node_, PNode):
                return f'{id(node_)}("{node_.__class__.__name__}")'
            return f'{id(node_)}["{node_.__class__.__name__}"]'

        lines = [f"```mermaid\nflowchart {direction}"]
        visited = set()
        stack = [self.start]

        while stack:
            curr_node = stack.pop()
            if curr_node not in visited:
                visited.add(curr_node)
                for action, next_node in curr_node._action_node_dict.items():
                    l_node = _node_to_str(curr_node)
                    r_node = _node_to_str(next_node)
                    if action == Node.DEFAULT:
                        lines.append(f'{l_node} --> {r_node}')
                    else:
                        lines.append(f'{l_node} -- "{action}" --> {r_node}')
                    stack.append(next_node)

        if len(visited) == 1:
            lines.append(_node_to_str(self.start))

        lines.append("```")
        display(Markdown("\n".join(lines)))


class Flow(FlowABC):
    """Create a flow class for synchronous execution.
    """

    def __call__(
            self,
            **kwargs,
    ) -> FlowABC:
        """Invoke the flow synchronously.
        """
        self.observer.on_flow_start(self)
        try:
            self._put_kwargs(**kwargs)
            if self._curr is None:
                self._curr = self.start
            while self._curr:
                self._prev = self._curr
                self.observer.on_node_start(self._curr)
                try:
                    self._curr = self._curr(self.scratchpad)
                finally:
                    self.observer.on_node_end(self._prev)
        finally:
            self.observer.on_flow_end(self)
            return self


class AFlow(FlowABC):
    """Create a flow class for asynchronous execution.
    """

    async def __call__(
            self,
            **kwargs,
    ) -> FlowABC:
        """Invoke the flow asynchronously.
        """
        self.observer.on_flow_start(self)
        try:
            self._put_kwargs(**kwargs)
            if self._curr is None:
                self._curr = self.start
            while self._curr:
                self._prev = self._curr
                self.observer.on_node_start(self._curr)
                try:
                    self._curr = await self._curr(self.scratchpad)
                finally:
                    self.observer.on_node_end(self._prev)
        finally:
            self.observer.on_flow_end(self)
            return self
