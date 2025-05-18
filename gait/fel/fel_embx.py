from typing import List

from .fel_base import FELLine
from .fel_vss import FELVSS
from ..flow import Node
from ..scratchpad import Scratchpad


class FELEmbX(Node):

    def __init__(
            self,
            vss: FELVSS,
            num_vec: int = 20,
    ) -> None:
        """Initialize the FELEmbX class.

        :param vss: The VSS (Vector Store Search) object used for embedding.
        :param num_vec: The number of embeddings to retrieve.
        """
        super().__init__()
        self.vss = vss
        self.num_vec = num_vec

    @property
    def sp_key(self) -> str:
        """Return the key used to store the results in the Scratchpad.
        """
        return ""

    def query(self, prompt: str) -> List[FELLine]:
        """Query the VSS for embeddings based on the provided prompt.

        :param prompt: The input prompt to query the VSS.
        :return: A list of FELLine objects representing the results.
        """
        return []

    def exec(self, sp: Scratchpad) -> str:
        """Execute the embedding process.

        :param sp: The Scratchpad object containing the input prompt.
        """
        sp[self.sp_key] = self.query(sp["prompt"])
        return Node.DEFAULT
