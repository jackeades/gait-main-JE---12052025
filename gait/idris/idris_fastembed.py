from typing import List, Sequence

from .idris_base import IdrisException
from .idris_emb import IdrisEmbInMemory


class IdrisFastembed(IdrisEmbInMemory):
    def __init__(
            self,
            # https://qdrant.github.io/fastembed/examples/Supported_Models/#supported-text-embedding-models
            model_name: str = "BAAI/bge-base-en-v1.5",
            max_context: int = 5,
            max_question_sql: int = 5,
            **kwargs,
    ) -> None:
        super().__init__(max_context, max_question_sql)
        try:
            from fastembed import TextEmbedding
        except ImportError:
            raise IdrisException("Please run `pip install fastembed` to use IdrisFastembed.")
        self._TextEmbedding = TextEmbedding
        self.text_emb = TextEmbedding(model_name=model_name, **kwargs)
        self.kwargs = kwargs

    def _get_embedding(self, prompt: str) -> Sequence[float]:
        return next(iter(self.text_emb.embed(prompt)))

    def _get_embeddings(self, prompts: List[str]) -> List[Sequence[float]]:
        return list(self.text_emb.embed(prompts))
