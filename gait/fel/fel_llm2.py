from textwrap import dedent

from litellm import completion

from .fel_base import FEL, FEL2
from .fel_observer import FELObserverABC, FELObserverNoop
from ..flow import Node
from ..scratchpad import Scratchpad
from ..utils import u_message, s_message, a_message

FELLLM2_SYSTEM = dedent(
    """
You are a highly skilled AI assistant designed to analyze user queries and accurately extract entities as a JSON object.

The user will ask questions related to one or more of the following geospatial layers.
Use the provided schema definitions to identify the correct layer and map the user’s request to a valid JSON object.
Ensure your response is concise and emits only the JSON result.
Accuracy is CRITICAL.

Geospatial Layers and Schemas:
{attributes}

The spatial relations between geospatial layers are:
- intersects
- notIntersects
- withinDistance
- notWithinDistance
- contains
- notContains
- within
- notWithin

Instructions:
- Understand the user query and identify the relevant geospatial layer(s) and attribute(s) based on the schema provided.
- Construct a correct JSON object from the extracted entities.
- Output only the JSON result, with no additional formatting, explanations, or tags.
- Ensure your answer is 100% accurate, as a reward of $1,000,000 is contingent upon correctness.
{instructions}
"""
).strip()


class FELLLM2(Node):
    def __init__(self, observer: FELObserverABC = None) -> None:
        super().__init__()
        self.observer = observer or FELObserverNoop()

    def exec(self, sp: Scratchpad) -> str:
        layers = sp["layers"]
        llm0 = sp["llm0"]
        fel = FEL(layers=[_ for _ in layers if _.name in (llm0.layer1, llm0.layer2)])

        messages = [
            s_message(FELLLM2_SYSTEM.format(
                attributes=fel.attributes(),
                instructions=sp.get("instructions", "")
            ))
        ]
        for fel_line in sp["fel2"]:
            self.observer.on_fel_line(fel_line)
            messages.append(u_message(fel_line.line))
            messages.append(a_message(fel_line.fel.model_dump_json()))

        messages.append(u_message(sp["prompt"]))
        response = completion(
            model=sp["model"],
            api_base=sp.get("api_base", None),
            api_version=sp.get("api_version", None),
            response_format=FEL2,
            temperature=0.0,
            messages=messages,
        )
        # logging.info(response.choices[0].message.content)
        llm2 = FEL2.model_validate_json(response.choices[0].message.content)
        self.observer.on_fel2(llm2)
        sp["fel"] = llm2
        return Node.DEFAULT
