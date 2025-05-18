import random

import litellm
from faker import Faker
from rich.pretty import pprint

from gait import Agent, MAO, ObserverLoguru

enable_langfuse = False
if enable_langfuse:
    litellm.success_callback = ["langfuse"]
    litellm.error_callback = ["langfuse"]


def get_temperature_at_location(location: str) -> dict:
    """Get temperature in Fahrenheit at a location. A location can be a city, a country, or a region.

    :param location: The location where to get the temperature.
    """
    return {
        "location": location,
        "temperature": int(random.uniform(-5, 40) * 10.0) / 10.0,
        "unit": "F",
    }


if __name__ == "__main__":
    fake = Faker()  # Help with creating fake city names.

    # Create a new agent with a function to get the temperature at a location.
    agent = Agent(
            functions=[get_temperature_at_location],
        # This is the LLM model temperature :-)
        temperature=0.0,
    )

    # Create a multi-agent orchestrator. Here we only have one agent.
    mao = MAO(agent, observer=ObserverLoguru())

    # Start the MAO loop.
    for _ in mao(f"Get the temperature in {fake.city()}.", iterations=3):
        # Check if we got a response.
        if _.content:
            # If so, stop the loop and print the response.
            mao.terminate()
            print(_.content)

    # Print the dialog or the list of messages exchanged with the LLM.
    for _ in mao.dialog:
        pprint(_, expand_all=True)
