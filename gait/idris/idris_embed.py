from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np
from numpy import ndarray

from .idris_base import IdrisException


class IdrisEmbed(ABC):

    @abstractmethod
    def indices(
            self,
            embeddings: Iterable[ndarray],
            embedding: ndarray,
    ) -> ndarray:
        """Get the indices for an embedding by descending order.

        :param embeddings: The embeddings to get the indices from.
        :param embedding: The embedding to get the indices for.
        :return: The indices for the prompt.
        """

    @abstractmethod
    def embedding(self, prompt: str) -> ndarray:
        """Get the embedding for a single prompt.

        :param prompt: The prompt to get the embedding for.
        :return: The embedding for the prompt.
        """

    @abstractmethod
    def embeddings(self, prompts: Iterable[str]) -> Iterable[ndarray]:
        """Get the embeddings for a list of prompts.

        :param prompts: The prompts to get the embeddings for.
        :return: The embeddings for the prompts.
        """


class IdrisTextEmbedding(IdrisEmbed):
    def __init__(
            self,
            model_name: str = "BAAI/bge-base-en-v1.5",
            **kwargs,
    ) -> None:
        """Initialize the IdrisTextEmbedding.

        :param model_name: The model name to use. Default is "BAAI/bge-base-en-v1.5".
        :param kwargs: Additional arguments for the model.
        """
        try:
            from fastembed import TextEmbedding
        except ImportError:
            raise IdrisException("Please run `pip install \".[fastembed]\"` to use IdrisTextEmbedding.")
        self.text_emb = TextEmbedding(
            model_name=model_name,
            **kwargs,
        )

    def indices(
            self,
            embeddings: Iterable[ndarray],
            embedding: ndarray,
    ) -> ndarray:
        """Get the indices for an embedding in descending order.

        :param embeddings: The embeddings to get the indices from.
        :param embedding: The embedding to get the indices for.
        :return: The indices for the prompt.
        """
        # embeddings_array = np.array(embeddings)
        # # Compute cosine similarity
        # norm_embeddings = np.linalg.norm(embeddings_array, axis=1)
        # norm_embedding = np.linalg.norm(embedding)
        # similarities = np.dot(embeddings_array, embedding) / (norm_embeddings * norm_embedding)
        similarities = np.dot(embeddings, embedding)
        return np.argsort(similarities)[::-1]

    def embedding(
            self,
            prompt: str
    ) -> ndarray:
        """Get the embedding for a single prompt.

        :param prompt: The prompt to get the embedding for.
        :return: The embedding for the prompt.
        """
        return next(iter(self.text_emb.embed(prompt)))

    def embeddings(
            self,
            prompts: Iterable[str]
    ) -> Iterable[ndarray]:
        """Get the embeddings for a list of prompts.

        :param prompts: The prompts to get the embeddings for.
        :return: The embeddings for the prompts.
        """
        return self.text_emb.embed(prompts)
