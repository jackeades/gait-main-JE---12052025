#
# https://anderfernandez.com/en/blog/chroma-vector-database-tutorial/
#
import json
import logging
import os
from dataclasses import dataclass
from typing import List, Any, Mapping, Callable, NamedTuple

from .fel_base import FELLine, FEL0, FEL1, FEL2
from .fel_vss import FELVSS

try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError as e:
    class chromadb(NamedTuple):
        Collection: List


@dataclass
class FELChroma(FELVSS):
    """Create VSS using ChromaDB.
    This class is responsible for creating and managing the Vector Store and Search (VSS) using ChromaDB.

    :param path: The path to the ChromaDB database. If "memory", it will use an in-memory database.
    :param model_name: The name of the embedding model to use. Default is "multi-qa-mpnet-base-cos-v1".
    :param device: The device to use for the embedding model (e.g., "cpu", "cuda"). Default is "cpu".
    :param fel0_coll_name: The name of the collection for FEL0 embeddings. Default is "fel0".
    :param fel1_coll_name: The name of the collection for FEL1 embeddings. Default is "fel1".
    :param fel2_coll_name: The name of the collection for FEL2 embeddings. Default is "fel2".
    """
    path: str = "memory"
    model_name: str = "multi-qa-mpnet-base-cos-v1"
    device: str = "cpu"
    fel0_coll_name: str = "fel0"
    fel1_coll_name: str = "fel1"
    fel2_coll_name: str = "fel2"

    def __post_init__(self) -> None:
        self.path = os.path.expanduser(self.path)
        self.client = chromadb.Client() if self.path == "memory" else chromadb.PersistentClient(path=self.path)
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.model_name,
            device=self.device,
            normalize_embeddings=True,
        )
        self.fel0_coll = self._get_collection(self.fel0_coll_name)
        self.fel1_coll = self._get_collection(self.fel1_coll_name)
        self.fel2_coll = self._get_collection(self.fel2_coll_name)

    def _get_collection(self, name: str) -> chromadb.Collection:
        return self.client.get_or_create_collection(
            name=name,
            embedding_function=self.ef,
            metadata={
                "model": self.model_name,
                "hnsw:space": "ip",  # Inner product because of normalized embeddings.
            },
        )

    def _add_collection(
            self,
            collection: chromadb.Collection,
            documents: List[str],
            metadatas: List[str | Mapping[str, Any]],
    ) -> None:
        ids = [str(index) for index in range(len(documents))]
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )

    def _create_fel(
            self,
            fel_lines: List[FELLine],
            collection: chromadb.Collection,
    ) -> None:
        documents = [_.line for _ in fel_lines]
        metadatas = [_.fel.model_dump() for _ in fel_lines]
        self._add_collection(
            collection=collection,
            documents=documents,
            metadatas=metadatas,
        )

    def create_fel0(self, fel_lines: List[FELLine]) -> None:
        """Create FEL0 embeddings and add them to the collection.

        :param fel_lines: A list of FELLine objects to be added to the collection.
        """
        self._create_fel(fel_lines, self.fel0_coll)

    def create_fel1(self, fel_lines: List[FELLine]) -> None:
        """Create FEL1 embeddings and add them to the collection.

        :param fel_lines: A list of FELLine objects to be added to the collection.
        """
        self._create_fel(fel_lines, self.fel1_coll)

    def create_fel2(self, fel_lines: List[FELLine]) -> None:
        """Create FEL2 embeddings and add them to the collection.

        :param fel_lines: A list of FELLine objects to be added to the collection.
        """
        self._create_fel(fel_lines, self.fel2_coll)

    def _query_fel(
            self,
            query: str,
            n_results: int,
            collection: chromadb.Collection,
            func: Callable,
    ) -> List[FELLine]:
        resp = collection.query(
            query_texts=query,
            n_results=n_results,
            include=["documents", "metadatas"],
        )
        documents = resp["documents"][0]
        metadatas = resp["metadatas"][0]
        return [FELLine(line=d, fel=func(m)) for d, m in zip(documents, metadatas)]

    def query_fel0(
            self,
            query: str,
            n_results: int = 10,
    ) -> List[FELLine]:
        """Query the VSS for FEL0 embeddings based on the provided query.

        :param query: The input query to search for.
        :param n_results: The number of results to return.
        """
        return self._query_fel(
            query,
            n_results,
            self.fel0_coll,
            FEL0.model_validate,
        )

    def query_fel1(
            self,
            query: str,
            n_results: int = 10,
    ) -> List[FELLine]:
        """Query the VSS for FEL1 embeddings based on the provided query.

        :param query: The input query to search for.
        :param n_results: The number of results to return.
        """
        return self._query_fel(
            query,
            n_results,
            self.fel1_coll,
            FEL1.model_validate,
        )

    def query_fel2(
            self,
            query: str,
            n_results: int = 10,
    ) -> List[FELLine]:
        """Query the VSS for FEL2 embeddings based on the provided query.

        :param query: The input query to search for.
        :param n_results: The number of results to return.
        """
        return self._query_fel(
            query,
            n_results,
            self.fel2_coll,
            FEL2.model_validate,
        )

    def _load(self, collection: chromadb.Collection, path: str, suffix: str) -> None:
        base, _ = os.path.splitext(path)
        filename = f"{base}{suffix}.json"
        if os.path.exists(filename):
            resp = collection.get(include=[])
            if resp["ids"]:
                collection.delete(resp["ids"])
            with open(filename, mode="r", encoding="utf-8") as fp:
                docs = json.load(fp)
            ids = [_["id"] for _ in docs]
            documents = [_["text"] for _ in docs]
            metadatas = [_["meta"] for _ in docs]
            collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        else:
            logging.warning(f"File {filename} not found. Skipping load.")

    def load(self, filename=None) -> None:
        """Load JSON files into the collections.

        :param filename: The base filename to use for the input files. If None, the default path is used.
        """
        path = filename or self.path
        self._load(self.fel0_coll, path, "0")
        self._load(self.fel1_coll, path, "1")
        self._load(self.fel2_coll, path, "2")

    def _dump(
            self,
            collection: chromadb.Collection,
            path: str,
            suffix: str,
    ) -> None:
        resp = collection.get()
        iden = resp["ids"]
        text = resp["documents"]
        meta = resp["metadatas"]
        base, ext = os.path.splitext(path)
        docs = [{"id": i, "text": t, "meta": m} for i, t, m in zip(iden, text, meta)]
        with open(f"{base}{suffix}.json", mode="w", encoding="utf-8") as fp:
            json.dump(docs, fp, ensure_ascii=False, indent=2)

    def dump(self, filename=None) -> None:
        """Create JSON files of the collections.

        :param filename: The base filename to use for the output files. If None, the default path is used.
        """
        if self.path == "memory" and filename is None:
            raise ValueError("Cannot dump memory path. Please provide a filename.")
        path = filename or self.path
        self._dump(self.fel0_coll, path, "0")
        self._dump(self.fel1_coll, path, "1")
        self._dump(self.fel2_coll, path, "2")
