from typing import List

from .fel_base import FELLine
from .fel_embx import FELEmbX


class FELEmb1(FELEmbX):
    @property
    def sp_key(self) -> str:
        """Return the key used to store the results in the Scratchpad.
        """
        return "fel1"

    def query(self, prompt: str) -> List[FELLine]:
        """Query the VSS for embeddings based on the provided prompt.

        :param prompt: The input prompt to query the VSS.
        :return: A list of FELLine objects representing the results.
        """
        return self.vss.query_fel1(prompt, self.num_vec)
