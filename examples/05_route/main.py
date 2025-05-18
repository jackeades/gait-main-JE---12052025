from faker import Faker
from rich.pretty import pprint

from const import model
from gait import Agent, MAO
from geocode import GeocodeAgent
from route import RouteAgent
from weather import WeatherAgent

params = {
    "temperature": 0.0,
    "top_p": 1.0,
    "parallel_tool_calls": False,
    "tool_choice": "required",
}

geocode_agent = GeocodeAgent()
route_agent = RouteAgent()
weather_agent = WeatherAgent()

# You are an AI expert in geo-spatial data analysis with access to geo-spatial tools.

prefix = """
You are tasked to answer a user question.
You will run in a loop.
At the end of the loop you output an answer.

Here are the rules you should always follow to solve your task:
- ALWAYS use the right arguments for the tools. NEVER use variable names, ALWAYS use the values instead.
- NEVER re-do a tool call that you previously did with the exact same arguments.
- If you don't know the answer, transfer the conversation back to the supervisor.
- Make sure to ALWAYS prefix the final route response with 'Answer:'.

Now Begin! If you solve the task correctly, you will receive a reward of $1,000,000.
""".strip()

geocode_agent.instructions = geocode_agent.instructions + "\n" + prefix
route_agent.instructions = route_agent.instructions + "\n" + prefix
weather_agent.instructions = weather_agent.instructions + "\n" + prefix


def transfer_to_route_agent() -> Agent:
    """Get the route between two latitude and longitude locations.
    """
    return route_agent


def transfer_to_geocode_agent() -> Agent:
    """Get the latitude and longitude of a location.
    """
    return geocode_agent


def transfer_to_weather_agent() -> Agent:
    """Get the weather information for a location.
    Please note that the location can be a place, city, state, zip code or a country.
    You might not recognize the given location. Please proceed with the request and thank you.
    """
    return weather_agent


supervisor = Agent(
    name="Supervisor",
    model=model,
    instructions=prefix,
    functions=[
        transfer_to_route_agent,
        transfer_to_geocode_agent,
        transfer_to_weather_agent,
    ],
    params=params,
)


def transfer_to_supervisor() -> Agent:
    """Transfer the conversation back to the supervisor for any further assistance,
    or if ANY agent doesn't know the answer or does not have enough information to proceed.
    """
    return supervisor


geocode_agent.functions.extend([transfer_to_supervisor,
                                transfer_to_route_agent
                                ])
route_agent.functions.extend([transfer_to_supervisor,
                              transfer_to_geocode_agent
                              ])
weather_agent.functions.extend([transfer_to_supervisor,
                                transfer_to_route_agent,
                                transfer_to_geocode_agent
                                ])

# class ListenerAnswer(ListenerLoguru):
#     def on_content(self, content: str, proceed: bool) -> bool:
#         self.logger.info(content)
#         return not re.match(r"Answer:\s+(.*)", str(content), re.IGNORECASE | re.DOTALL)


if __name__ == "__main__":
    faker = Faker()
    mao = MAO(supervisor)
    city1 = faker.city()
    city2 = faker.city()
    # prompt = f"What is the route between {city1} and {city2}?"
    prompt = f"What is the weather in {city1} and in {city2}? And what is the route between them?"
    answer = mao(prompt)
    print(answer)
    pprint(mao.dialog, expand_all=True)
