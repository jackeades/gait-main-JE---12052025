#
# https://docs.litellm.ai/docs/
#
import logging
from typing import List, Optional, Literal, Dict, Sequence

import litellm
from litellm import completion, embedding

from .idris_base import IdrisException, IdrisLLM
from .idris_emb import IdrisEmbInMemory


class IdrisLiteLLMException(IdrisException):
    def __init__(self) -> None:
        super().__init__("Please run `pip install litellm` to use IdrisLiteLLM.")


class IdrisLiteLLM(IdrisLLM):
    def __init__(
            self,
            model_name: str = "openai/gpt-4o-mini",
            temperature: float = 0.0,
            **kwargs,
    ) -> None:
        """
        Initialize a language model that uses the LiteLLM API.

        :param model_name: The name of the model to use. Default: "gpt-4o"
        :param temperature: The temperature to use. Default: 0.0.
        :param kwargs: Additional LiteLLM keyword arguments.
        """
        if litellm is None:
            raise IdrisLiteLLMException()

        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_name = model_name
        self.temperature = temperature
        self.kwargs = kwargs

    def close(self) -> None:
        return

    @property
    def client(self):
        return None

    def create_message(
            self,
            role: Literal["user", "assistant", "system", "tool"],
            content: str
    ) -> Dict[str, str]:
        return {"role": role, "content": content}

    def generate_sql(
            self,
            messages: List[Dict[str, str]],
            **kwargs
    ) -> Optional[str]:
        params = {
            "model": self.model_name,
            "temperature": self.temperature,
            "messages": messages,
            **self.kwargs,
            **kwargs,
        }
        response = completion(**params)
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(response)
        return response.choices[0].message.content


class IdrisLiteEmb(IdrisEmbInMemory):
    def __init__(
            self,
            model_name: str = "openai/text-embedding-ada-002",
            max_context: int = 5,
            max_question_sql: int = 5,
            **kwargs
    ) -> None:
        """
        Initialize an in-memory embedding agent that uses the LiteLLM API or AzureLiteLLM API.
        If api_version is None, the LiteLLM API is used. Otherwise, the AzureLiteLLM API is used.

        :param model_name: The name of the model to use. Default: "text-embedding-ada-002"
        :param max_context:  The maximum number of context embeddings to store. Default: 5
        :param max_question_sql: The maximum number of question SQL embeddings to store. Default: 5
        :param kwargs: Additional keyword arguments.
        """
        super().__init__(max_context, max_question_sql)
        if litellm is None:
            raise IdrisLiteLLMException()
        self.model_name = model_name
        self.kwargs = kwargs

    @property
    def client(self):
        return None

    def close(self) -> None:
        return None

    def _get_embedding(self, prompt: str) -> Sequence[float]:
        resp = embedding(input=[prompt], model=self.model_name, **self.kwargs)
        return resp.data[0]["embedding"]

    def _get_embeddings(self, prompts: List[str]) -> List[Sequence[float]]:
        resp = embedding(input=prompts, model=self.model_name, **self.kwargs)
        return [d["embedding"] for d in resp.data]
