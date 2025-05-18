import inspect
import re
import types
import typing
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type

from .scratchpad import __SCRATCHPAD__
from .types import MessageRole, MessageType


def s_message(content: str) -> MessageType:
    """Create a system message.

    :param content: The content of the message.
    :return: A system message.
    """
    return dict(role=MessageRole.SYSTEM.value, content=content)


def u_message(content: str) -> MessageType:
    """Create a user message.

    :param content: The content of the message.
    :return: A user message.
    """
    return dict(role=MessageRole.USER.value, content=content)


def a_message(content: str) -> MessageType:
    """Create an assistant message.

    :param content: The content of the message.
    :return: An assistant message.
    """
    return dict(role=MessageRole.ASSISTANT.value, content=content)


def t_message(content: str, name: str, tool_call_id: str) -> MessageType:
    """Create a tool message.

    :param content: The content of the message.
    :param name: The name of the tool.
    :param tool_call_id: The tool call ID.
    :return: A tool message.
    """
    return dict(role=MessageRole.TOOL.value, content=content, name=name, tool_call_id=tool_call_id)


def _annotation_to_type(t: type) -> dict:
    if t == str:
        return {"type": "string"}
    elif t == int:
        return {"type": "integer"}
    elif t == float or t == int | float:
        return {"type": "number"}
    elif t == bool:
        return {"type": "boolean"}
    elif t == type(None):
        return {"type": "null"}
    elif t == list or t == List:
        return {"type": "array", "items": {}}
    elif t == dict or t == Dict:
        return {"type": "object", "additionalProperties": False}
    elif type(t) == types.GenericAlias:
        if t.__origin__ == list:
            return {"type": "array", "items": _annotation_to_type(t.__args__[0])}
        elif t.__origin__ == dict:
            if t.__args__[0] != str:
                raise ValueError(f"Unsupported type (JSON keys must be strings): {t}")
            return {
                "type": "object",
                "patternProperties": {".*": _annotation_to_type(t.__args__[1])},
            }
        else:
            raise ValueError(f"Unsupported type: {t}")
    elif type(t) == typing._LiteralGenericAlias:
        for arg in t.__args__:
            if type(arg) != type(t.__args__[0]):
                raise ValueError(f"Unsupported type (definite type is required): {t}")
        return {**_annotation_to_type(type(t.__args__[0])), "enum": t.__args__}
    else:
        raise ValueError(f"Unsupported type: {t}")


@dataclass
class FuncParameter:
    """Data class to represent a function parameter.
    """

    name: str
    annotation: Type[Any]
    description: Optional[str] = None
    required: bool = True
    default: Optional[any] = None

    def to_property(
            self,
    ) -> Dict[str, Any]:
        description = (
            self.description if self.description else self.name.replace("_", " ")
        )
        type_desc = _annotation_to_type(self.annotation)
        params = {"description": description, **type_desc}
        if self.default is not None:
            params["default"] = self.default
        return params


_PARAM_KEY: str = ":param"
_RETURN_KEY: str = ":return"
_DESCRIPTION_REGEX = rf"(?s)(.*?)(?=\n{_PARAM_KEY}|\n{_RETURN_KEY}|$)"
_PARAMS_REGEX = rf"(?s)({_PARAM_KEY} .*?: .*?)(?=\n{_PARAM_KEY}|\n{_RETURN_KEY}|$)"
_RETURN_REGEX = rf"({_RETURN_KEY}.*$)"
_RETURN_CONTENT = rf"{_RETURN_KEY}:\s*(.*)"


def function_to_tool(
        function: Callable,
        strict: bool = True,
) -> Dict:
    """Convert a function to a JSON-serializable dictionary that describes the function's signature.

    :param function: The function to be converted.
    :param strict: Whether to enforce strict typing.
    """
    doc = inspect.getdoc(function)
    sig = inspect.signature(function)

    func_name = function.__name__
    description = re.match(_DESCRIPTION_REGEX, doc).group(1).strip() if doc else ""
    if not description:
        description = func_name.replace("_", " ")

    params = (
        [param.group().strip() for param in re.finditer(_PARAMS_REGEX, doc)]
        if doc
        else []
    )
    name_desc = {}
    for param in params:
        param = param.replace(_PARAM_KEY, "").strip()
        name, desc = param.split(":")
        name_desc[name] = desc.strip()

    parameters = []
    for name, param in sig.parameters.items():
        if name not in ["self", "cls", __SCRATCHPAD__]:
            parameters.append(
                FuncParameter(
                    name,
                    param.annotation,
                    name_desc.get(name, name.replace("_", " ")),
                    param.default is inspect.Parameter.empty,
                    (
                        param.default
                        if param.default is not inspect.Parameter.empty
                        else None
                    ),
                )
            )
    properties = {p.name: p.to_property() for p in parameters}
    parameters = {
        "properties": properties,
        "required": [p.name for p in parameters if p.required],
        "description": description,
        "type": "object",
        "additionalProperties": False,
    }
    function = {
        "name": func_name,
        "parameters": parameters,
    }
    if strict:
        function["strict"] = True
    return {
        "type": "function",
        "function": function,
    }
