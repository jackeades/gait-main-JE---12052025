import random

from faker import Faker

from const import model
from gait import Agent


class WeatherAgent(Agent):
    def __init__(self):
        super().__init__(
            name="Weather Agent",
            model=model,
            instructions="You are a weather agent. You can help with weather information.",
            functions=[self.get_weather],
        )
        self.faker = Faker()

    def get_weather(self, location: str) -> dict:
        """Get the weather information for a location.

        :param location: The location to get the weather information for. Can be a place, city, state, zip code or a country.
        """
        return {
            "location": location,
            "temperature": self.faker.random_int(0, 40),
            "unit": random.choice(["C", "F"]),
            "condition": random.choice(["Sunny", "Cloudy", "Rainy", "Snowy"]),
        }
