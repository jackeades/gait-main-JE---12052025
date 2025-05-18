import json
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any

from litellm import completion

from .agent import Agent
from .dialog import Dialog
from .utils import s_message


class Operator(ABC):
    """Abstract class to implement an Operator.

    The Operator routes messages to agents just like a human switchboard operator.
    """

    @abstractmethod
    def route(self, dialog: Dialog) -> Optional[Agent]:
        """Route a message to an agent.

        :param dialog: The message log.
        :return: A reference to an agent.
        """

    @abstractmethod
    def close(self) -> None:
        """Close the router.

        Close open resources.
        """


_OPERATOR_SYSTEM_PROMPT = """
You are an agent matcher and an intelligent assistant designed to analyze user queries and match them with
the most suitable agent. Your task is to understand the user's request, identify key entities and intents, 
and determine which agent would be best equipped to handle the query.

Important: The user's input may be a follow-up response to a previous interaction.
The conversation history, including the name of the previously selected agent, is provided.
If the user's input appears to be a continuation of the previous conversation
(e.g., "yes", "ok", "I want to know more", "1"), select the same agent as before.

Analyze the user's input and select the correct agent_name:
<agents>
{AGENTS}
</agents>
If you are unable to select an agent, reply with "unknown".

Now Begin! If you solve the task correctly, you will receive a reward of $1,000,000.
"""

_OPERATOR_SYSTEM_PROMPT_1 = """
You are an Agent Matcher, an intelligent assistant designed to analyze user queries and direct them to the most suitable agent for response. Your task is to:
1.	Understand the user’s request by identifying key entities, intent, and context.
2.	Determine the most appropriate agent to handle the query from the given list of available agents.
3.	Ensure continuity in ongoing conversations by checking if the user’s input is a follow-up from a previous interaction.

Rules for Selecting the Correct Agent:
•	If the user’s input is a continuation of a previous conversation (e.g., “yes”, “ok”, “I want to know more”, “1”), assign the same agent as before. The conversation history, including the name of the last selected agent, is provided.
•	If the user initiates a new request, analyze the input and select the most relevant agent from the list below.
•	If no suitable agent is identified, respond with "unknown".

Agent List:

<agents>
{AGENTS}
</agents>

Now, analyze the user’s input and select the correct agent_name.

Begin! If you complete this task correctly, you will receive a reward of $1,000,000. 🚀
"""


class OperatorLiteLLM(Operator):
    def __init__(
            self,
            model: str,
            agents: List[Agent] = None,
            system_prompt: str = None,
            temperature: float = 0.0,
            params: Dict[str, Any] = None,
    ) -> None:
        """Initialize the Operator based on LiteLLM.

        :param model: The model to use.
        :param agents: The list of agents.
        :param system_prompt: Optional system prompt. Defaults to _ROUTE_SYSTEM_PROMPT.
        :param temperature: Optional temperature to use. Defaults to 0.0.
        :param params: Additional optional parameters to pass to the model.
        """
        if not model:
            raise ValueError("Model is required.")
        self.model = model
        self.agent_dict = {_agent.name: _agent for _agent in agents or []}
        self.system_prompt = system_prompt or _OPERATOR_SYSTEM_PROMPT
        self.temperature = temperature
        self.params = params or {}

    def _get_agents(self) -> str:
        return "\n".join([f"<agent_name>{_.name}</agent_name><agent_description>{_.description}</agent_description>"
                          for _ in
                          self.agent_dict.values()])

    def route(self, dialog: Dialog) -> Optional[Agent]:
        agents = self._get_agents()
        system_prompt = self.system_prompt.format(AGENTS=agents)
        messages = [s_message(system_prompt)]
        for message in dialog:
            messages.append(message)
        tools = [{
            "type": "function",
            "function": {
                "name": "transfer_to_agent",
                "description": "Transfer the conversation to an agent given an agent_name.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_name": {
                            "type": "string",
                            "description": "The name of the agent.",
                        },
                    },
                    "required": ["agent_name"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }]
        params = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": {"type": "function", "function": {"name": "transfer_to_agent"}},
            "temperature": self.temperature,
            **self.params,
        }
        response = completion(**params)
        agent = None
        for choice in response.choices:
            message = choice.message
            for tool in message.tool_calls or []:
                function = tool.function
                if function.name == "transfer_to_agent":
                    args = json.loads(function.arguments)
                    agent = self.agent_dict.get(args["agent_name"])
                    break
            break
        return agent

    def close(self) -> None:
        """Close the router.

        Close open resources.
        """
        return
