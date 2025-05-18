from textwrap import dedent

from litellm import completion

from .fel_base import FEL, FEL1
from .fel_observer import FELObserverABC, FELObserverNoop
from ..flow import Node
from ..scratchpad import Scratchpad
from ..utils import u_message, s_message, a_message

FELLLM1_SYSTEM = dedent(
    """
You are an expert AI assistant engineered to interpret user queries and extract entities into accurate JSON objects based on geospatial layer schemas.

The user will submit queries involving one or more of the following geospatial layers.
Leverage the provided schema definitions to pinpoint the appropriate layer(s) and transform the user’s request into a valid JSON object.
Your response must be succinct, delivering only the JSON output, with flawless accuracy as the top priority.

Geospatial Layers and Schemas:
{attributes}

Instructions:
- Analyze the user query to identify the relevant geospatial layer(s) and attribute(s) according to the schema.
- Build a precise JSON object that fully captures the extracted entities.
- Provide only the JSON object—no extra text, formatting, or commentary.
- Guarantee complete accuracy in entity extraction and JSON construction, as correctness is essential.
{instructions}
"""
).strip()


class FELLLM1(Node):
    def __init__(self, observer: FELObserverABC = None) -> None:
        super().__init__()
        self.observer = observer or FELObserverNoop()

    def exec(self, sp: Scratchpad) -> str:
        layers = sp["layers"]
        llm0 = sp["llm0"]
        fel = FEL(layers=[_ for _ in layers if _.name == llm0.layer1])

        messages = [
            s_message(FELLLM1_SYSTEM.format(
                attributes=fel.attributes(),
                instructions=sp.get("instructions", "")
            ))
        ]
        for fel_line in sp["fel1"]:
            self.observer.on_fel_line(fel_line)
            messages.append(u_message(fel_line.line))
            messages.append(a_message(fel_line.fel.model_dump_json()))

        messages.append(u_message(sp["prompt"]))
        response = completion(
            model=sp["model"],
            api_base=sp.get("api_base", None),
            api_version=sp.get("api_version", None),
            response_format=FEL1,
            temperature=0.0,
            messages=messages,
        )
        fel1 = FEL1.model_validate_json(response.choices[0].message.content)
        self.observer.on_fel1(fel1)
        sp["fel"] = fel1
        return Node.DEFAULT
