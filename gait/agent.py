import json
from dataclasses import dataclass, field
from typing import List, Callable, Dict, Any, Optional

from litellm import completion
from pydantic import BaseModel

from .dialog import Dialog
from .observer import Observer, ObserverNoop
from .scratchpad import Scratchpad, ScratchpadInMemory, __SCRATCHPAD__
from .types import InstructionType
from .utils import function_to_tool, t_message, s_message


@dataclass
class MCPServer:
    """MCPServer class to handle the server connection.

    :param url: The URL of the server.
    :param port: The port of the server.
    """
    url: str
    username: Optional[str] = None
    password: Optional[str] = None


@dataclass
class AgentResponse:
    """Agent response.

    :param finish_reason: The reason why the agent finishes the conversation. like stop, tool_call, etc..
    :param content: The content of the agent response.
    :param agent: Optional agent to transfer the conversation to.
    """
    finish_reason: Optional[str] = None
    content: Optional[str] = None
    agent: Optional["Agent"] = None


@dataclass
class Agent:
    """An agent that can perform functions.

    :param name: The name of the agent.
    :param description: The description of the agent.
    :param model: The model to use for the agent. Default is 'ollama_chat/llama3.2:latest'.
    :param instructions: The instructions to the agent. This is the system prompt. This can be a string or a callable.
    :param functions: The functions that the agent can use.
    :param params: Optional extra LiteLLM parameters to pass to the model completion.
    """
    name: str = "Agent"
    description: str = "You are a helpful AI agent."
    model: str = "ollama_chat/llama3.2:latest"
    # If instructions is a callable, then the function should accept a scratchpad as an argument.
    instructions: Optional[InstructionType] = None
    functions: List[Callable] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)

    def __init__(
            self,
            name: str = "Agent",
            description: str = "You are a helpful AI agent.",
            model: str = "ollama_chat/llama3.2:latest",
            instructions: Optional[InstructionType] = None,
            functions: List[Callable] = None,
            **kwargs,
    ) -> None:
        """Initialize the agent with kwargs.

        :param name: The name of the agent.
        :param description: The description of the agent. This is the system prompt when the agent is started.
        :param model: The model to use for the agent. Default is 'ollama_chat/llama3.2:latest'.
        :param instructions: The instructions to show when the agent is started. This can be a string or a callable.
        :param functions: The functions that the agent can use.
        :param kwargs: Optional extra LLMLite parameters to pass to the model.
        """
        self.name = name
        self.description = description
        self.model = model
        self.instructions = instructions if instructions is not None else description
        self.functions = functions if functions is not None else []
        self.params = kwargs or {}

    def __post_init__(self):
        """Convert the description to instructions if instructions is None.
        """
        if self.instructions is None:
            self.instructions = self.description

    def __call__(
            self,
            dialog: Dialog | str,
            scratchpad: Scratchpad = ScratchpadInMemory(),
            observer: Observer = ObserverNoop(),
    ) -> AgentResponse:
        """Process the dialog and return an agent response.

        :param dialog: An instance of Dialog or a string.
        :param scratchpad: The scratchpad to use for the agent. Default is an in-memory scratchpad.
        :param observer: The observer to use for the agent. Default is a no-op observer.
        :return: The agent response.
        """
        # Create an of dialog if dialog is a string.
        _dialog = dialog if isinstance(dialog, Dialog) else Dialog.instance() + str(dialog)
        agent_response = AgentResponse(agent=self)
        # Execute the instructions with scratchpad if callable.
        content = self.instructions(scratchpad) if callable(self.instructions) else self.instructions
        # Create the messages
        messages = [s_message(content)]
        messages.extend(_dialog)
        # Convert the functions to tools.
        tools = [function_to_tool(_) for _ in self.functions]
        params = {
            "model": self.model,
            "messages": messages,
            "tools": tools or None,
            **self.params,
        }
        # Invoke the model.
        comp_resp = completion(**params)
        for choice in comp_resp.choices:
            agent_response.finish_reason = choice.finish_reason
            choice_message = choice.message
            # Append the choice message to the dialog.
            _dialog += choice_message
            observer.on_content(choice_message.content)
            agent_response.content = choice_message.content
            func_dict = {_.__name__: _ for _ in self.functions}
            # Invoke the functions if any.
            for tool_call in choice_message.tool_calls or []:
                observer.on_function(tool_call.function.name, tool_call.function.arguments)
                func_name = tool_call.function.name
                if func_name in func_dict:
                    func = func_dict[func_name]
                    args = json.loads(tool_call.function.arguments)
                    # If the function has a scratchpad argument, then add it to the arguments.
                    if __SCRATCHPAD__ in func.__code__.co_varnames:
                        args[__SCRATCHPAD__] = scratchpad
                    # Invoke the function and get the content.
                    # TODO: Handle the case when the function throw an exception.
                    content = func(**args)
                    # Update the content based on the type.
                    match content:
                        case bool() | int() | float() | str():
                            content = str(content)
                        case dict() | list():
                            content = json.dumps(content, ensure_ascii=False)
                        case Agent() as agent:
                            agent_response.agent = agent
                            content = f"Transfer to agent '{agent.name}'."
                        # Handle the case when the function returns a pydantic model.
                        case BaseModel() as model:
                            content = model.model_dump_json()
                        case _:
                            raise ValueError(f"Unsupported response type: {type(content)}")
                    observer.on_observation(content)
                    # Append the content to the dialog.
                    _dialog += t_message(
                        content,
                        func_name,
                        tool_call.id)
                else:
                    raise ValueError(f"Error: Function '{func_name}' is not found.")
            break  # Process the first choice only.

        return agent_response
