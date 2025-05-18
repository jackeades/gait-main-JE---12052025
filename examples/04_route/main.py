import os

from faker import Faker

from gait import Agent, MAO, ObserverLoguru

deployment = f"/{os.environ['AZURE_API_DEPLOYMENT']}"
model = "azure" + deployment
faker = Faker()


def geocode_location(location: str) -> dict:
    """Geocode a location to get its latitude and longitude.

    :param location: The location to geocode. Can be a place, city, state, zip code or a country.
    """
    return {
        "location": location,
        "latitude": float(faker.latitude()),
        "longitude": float(faker.longitude()),
    }


def get_route(
        lon1: float,
        lat1: float,
        lon2: float,
        lat2: float,
) -> dict:
    """Get the route between a starting latitude/longitude location and an ending latitude/longitude location.

    :param lon1: The route starting longitude.
    :param lat1: The route starting latitude.
    :param lon2: The route ending longitude.
    :param lat2: The route ending latitude.
    """
    return {"start_at": f"{lat1},{lon1}", "end_at": f"{lat2},{lon2}", "route": f"{lat1},{lon1} -> {lat2},{lon2}"}


def get_current_temperature(
        location: str,
        celsius_or_fahrenheit: str,
) -> dict:
    """Get the current temperature at a given location.

    :param location: The location to get the temperature for. Can be a place, city, state, zip code or a country.
    :param celsius_or_fahrenheit: The unit to return the temperature in. Can be "C" for Celsius or "F" for Fahrenheit.
    """
    return {
        "location": location,
        "temperature": faker.random_int(0, 100),
        "unit": celsius_or_fahrenheit,
    }


instructions = """
You are an AI expert in geo-spatial data analysis with access to geo-spatial tools.
You are tasked to answer a user question.
You will run in a loop.
At the end of the loop you output an answer.

Here are the rules you should always follow to solve your task:
- ALWAYS use the right arguments for the tools. NEVER use variable names, ALWAYS use the values instead.
- NEVER re-do a tool call that you previously did with the exact same arguments.

Now Begin! If you solve the task correctly, you will receive a reward of $1,000,000.
""".strip()

triage_agent = Agent(
    instructions=instructions,
    functions=[geocode_location, get_route, get_current_temperature],
    model=model,
    params={
        "api_key": os.environ["AZURE_API_KEY"],
        "api_base": os.environ["AZURE_API_ENDPOINT"] + deployment,
        "api_version": os.environ["AZURE_API_VERSION"],
        "temperature": 0.0,
        "top_p": 1.0,
    }
)

prompt = f"""
What's the current temperature in {faker.city()} in celsius and {faker.city()} in fahrenheit?
And what is the route between them?
""".strip()

if __name__ == "__main__":
    mao = MAO(triage_agent)
    mao(prompt)
    # pprint(llm.messages, expand_all=True)
