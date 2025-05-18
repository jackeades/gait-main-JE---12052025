from enum import Enum
from typing import Union, Callable, Dict

from .scratchpad import Scratchpad

MessageType = Dict[str, str]
InstructionType = Union[str, Callable[[Scratchpad], str]]


class MessageRole(Enum):
    """Message role.
    """
    SYSTEM = "system"
    ASSISTANT = "assistant"
    USER = "user"
    TOOL = "tool"
