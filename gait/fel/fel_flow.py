from datetime import datetime

from .fel_base import FEL1, FEL2, Layers
from .fel_emb0 import FELEmb0
from .fel_emb1 import FELEmb1
from .fel_emb2 import FELEmb2
from .fel_llm0 import FELLLM0
from .fel_llm1 import FELLLM1
from .fel_llm2 import FELLLM2
from .fel_memory import FELMemory
from .fel_observer import FELObserverABC, FELObserverNoop
from ..flow import Flow


class FELFlow:
    def __init__(
            self,
            fel_path: str,
            model: str = "azure/gpt-4o",
            api_base: str = None,
            api_version: str = None,
            device: str = "cpu",
            num_fel0: int = 20,
            num_fel1: int = 20,
            num_fel2: int = 20,
            observer: FELObserverABC = None,
    ) -> None:
        layers = Layers.load(fel_path).layers
        vss = FELMemory(fel_path=fel_path, device=device)

        if observer is None:
            observer = FELObserverNoop()

        emb0 = FELEmb0(vss, num_fel0)
        emb0 >> (llm0 := FELLLM0(observer=observer))
        llm0 - "1" >> FELEmb1(vss, num_fel1) >> FELLLM1(observer=observer)
        llm0 - "2" >> FELEmb2(vss, num_fel2) >> FELLLM2(observer=observer)

        self.layers = layers
        self.model = model
        self.api_base = api_base
        self.api_version = api_version
        self.emb0 = emb0

    def __call__(self, prompt: str) -> FEL1 | FEL2:
        now = datetime.now()
        today = now.strftime("%A %b %d, %Y")
        hh_mm_am_pm = now.strftime("%I:%M %p")
        flow = Flow(
            self.emb0,
        )
        flow(
            prompt=prompt,
            layers=self.layers,
            model=self.model,
            api_base=self.api_base,
            api_version=self.api_version,
            instructions="\n".join(
                [
                    f"- Today is {today}.",
                    f"- The time right now is {hh_mm_am_pm}.",
                ]
            ),
        )
        return flow["fel"]
