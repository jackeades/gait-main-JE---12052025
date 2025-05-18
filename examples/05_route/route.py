import random

from faker import Faker

from const import model
from gait import Agent


class RouteAgent(Agent):
    def __init__(self):
        super().__init__(
            name="Route Agent",
            model=model,
            instructions=" ".join([
                "You are a route agent.",
                "You can help with routing directions."
            ]),
            functions=[self.get_route],
        )
        self.faker = Faker()

    def get_route(
            self,
            lon1: float,
            lat1: float,
            lon2: float,
            lat2: float,
    ) -> dict:
        """Get the route between an origin latitude/longitude location and a destination latitude/longitude location.
        Just report the route as a text string WITHOUT any additional information.

        :param lon1: The route starting longitude.
        :param lat1: The route starting latitude.
        :param lon2: The route ending longitude.
        :param lat2: The route ending latitude.
        """
        segments = []
        for _ in range(random.randint(1, 5)):
            direction = random.choice(["turn left", "turn right", "go straight"])
            distance = random.randint(1, 10)
            segments.append(f"{direction} for {distance} miles.")
        return {
            "origin": f"{lat1},{lon1}",
            "destination": f"{lat2},{lon2}",
            "route": " ".join(segments),
        }
