from faker import Faker

from const import model
from gait import Agent


class GeocodeAgent(Agent):
    def __init__(self):
        super().__init__(
            name="Geocode Agent",
            model=model,
            instructions="You are a geocode agent. You can help with geocoding locations.",
            functions=[self.geocode_location],
        )
        self.faker = Faker()

    def geocode_location(self, location: str) -> dict:
        """Geocode a location to get its latitude and longitude.

        :param location: The location to geocode. Can be a place, city, state, zip code or a country.
        """
        return {
            "location": location,
            "latitude": float(self.faker.latitude()),
            "longitude": float(self.faker.longitude()),
        }
