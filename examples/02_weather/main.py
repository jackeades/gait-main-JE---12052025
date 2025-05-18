import random

from faker import Faker
from rich.pretty import pprint

from gait import Agent, MAO


def get_weather(location: str) -> dict:
    """Get the temperature in Fahrenheit and whether it's raining at a location.

    :param location: The location to get the temperature for. Can be a place, city, state, zip code or a country.
    """
    return {
        "location": location,
        "temperature": int(random.uniform(-5, 40) * 10.0) / 10.0,
        "unit": "F",
        "raining": random.choice(["Yes", "No"]),
    }


if __name__ == "__main__":
    faker = Faker()
    agent = Agent(
        functions=[get_weather],
        temperature=0.1,
    )
    mao = MAO(agent)
    city = faker.city()
    for _ in mao(f"Do I need an umbrella? I'm in {city}."):
        if _.content:
            mao.terminate()
            print(_.content)

    for _ in mao.dialog:
        pprint(_, expand_all=True)
