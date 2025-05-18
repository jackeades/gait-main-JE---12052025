import json
import os
from dataclasses import dataclass, field
from typing import List, Callable

import numpy as np
from sentence_transformers import SentenceTransformer, util

from .fel_base import FELLine, FEL0, FEL1, FEL2
from .fel_vss import FELVSS


@dataclass
class FELMemory(FELVSS):
    """Create In-Memory VSS.

    :param model_name: The name of the embedding model to use. Default is "multi-qa-mpnet-base-cos-v1".
    :param device: The device to use for the embedding model (e.g., "cpu", "cuda"). Default is "cpu".
    """
    model_name: str = "multi-qa-mpnet-base-cos-v1"
    device: str = "cpu"
    batch_size: int = 20
    show_progress_bar: bool = False
    overwrite_json_files: bool = False
    fel_path: str = None

    _fel0: List[FELLine] = field(default=None, init=False)
    _fel1: List[FELLine] = field(default=None, init=False)
    _fel2: List[FELLine] = field(default=None, init=False)
    _emb0: np.ndarray = field(default=None, init=False)
    _emb1: np.ndarray = field(default=None, init=False)
    _emb2: np.ndarray = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.model = SentenceTransformer(
            model_name_or_path=self.model_name,
            device=self.device,
        )
        if self.fel_path is not None:
            base, _ = os.path.splitext(self.fel_path)
            self._fel0 = self._load_json(f"{base}0.json", FEL0)
            self._fel1 = self._load_json(f"{base}1.json", FEL1)
            self._fel2 = self._load_json(f"{base}2.json", FEL2)
            npz = np.load(base + ".npz")
            self._emb0 = npz["emb0"]
            self._emb1 = npz["emb1"]
            self._emb2 = npz["emb2"]

    def _create_emb(
            self,
            fel_lines: List[FELLine],
    ) -> np.ndarray:
        sentences = [_.line for _ in fel_lines]
        return self.model.encode(
            sentences,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            normalize_embeddings=True,
        )

    def create_fel0(self, fel_lines: List[FELLine]) -> None:
        """Create FEL0 embeddings and add them to the collection.

        :param fel_lines: A list of FELLine objects to be added to the collection.
        """
        self._fel0 = fel_lines
        self._emb0 = self._create_emb(fel_lines)

    def create_fel1(self, fel_lines: List[FELLine]) -> None:
        """Create FEL1 embeddings and add them to the collection.

        :param fel_lines: A list of FELLine objects to be added to the collection.
        """
        self._fel1 = fel_lines
        self._emb1 = self._create_emb(fel_lines)

    def create_fel2(self, fel_lines: List[FELLine]) -> None:
        """Create FEL2 embeddings and add them to the collection.

        :param fel_lines: A list of FELLine objects to be added to the collection.
        """
        self._fel2 = fel_lines
        self._emb2 = self._create_emb(fel_lines)

    def query_fel0(
            self,
            query: str,
            n_results: int = 10,
    ) -> List[FELLine]:
        """Query the VSS for FEL0 embeddings based on the provided query.

        :param query: The input query to search for.
        :param n_results: The number of results to return.
        """
        embedding = self.model.encode([query], normalize_embeddings=True)
        return [
            self._fel0[index]
            for index in util.dot_score(
                self._emb0,
                embedding
            ).reshape(-1).argsort()[-n_results:]]

    def query_fel1(
            self,
            query: str,
            n_results: int = 10,
    ) -> List[FELLine]:
        """Query the VSS for FEL1 embeddings based on the provided query.

        :param query: The input query to search for.
        :param n_results: The number of results to return.
        """
        embedding = self.model.encode([query], normalize_embeddings=True)
        return [
            self._fel1[index]
            for index in util.dot_score(
                self._emb1,
                embedding
            ).reshape(-1).argsort()[-n_results:]]

    def query_fel2(
            self,
            query: str,
            n_results: int = 10,
    ) -> List[FELLine]:
        """Query the VSS for FEL2 embeddings based on the provided query.

        :param query: The input query to search for.
        :param n_results: The number of results to return.
        """
        embedding = self.model.encode([query], normalize_embeddings=True)
        return [
            self._fel2[index]
            for index in util.dot_score(
                self._emb2,
                embedding
            ).reshape(-1).argsort()[-n_results:]]

    def _load_json(self, filename: str, clazz: Callable) -> List[FELLine]:
        """Load a collection from a file.

        :param filename: The name of the file to load.
        :param clazz: The function to use for loading the collection.
        :return: A list of FELLine objects.
        """
        with open(filename, mode="r", encoding="utf-8") as fp:
            docs = json.load(fp)
            return [
                FELLine(
                    line=doc["text"],
                    fel=clazz(**doc["meta"]),
                )
                for doc in docs
            ]

    def load(self, filename: str) -> None:
        """Load JSON files and create the embeddings.

        :param filename: The base filename to use for the input files. If None, the default path is used.
        """
        base, _ = os.path.splitext(filename)
        self._fel0 = self._load_json(f"{base}0.json", FEL0)
        self._fel1 = self._load_json(f"{base}1.json", FEL1)
        self._fel2 = self._load_json(f"{base}2.json", FEL2)
        self._emb0 = self._create_emb(self._fel0)
        self._emb1 = self._create_emb(self._fel1)
        self._emb2 = self._create_emb(self._fel2)

    def _dump(self, fel_lines: List[FELLine], filename: str) -> None:
        """Dump the embeddings to a JSON file.

        :param fel_lines: A list of FELLine objects to be dumped.
        :param filename: The name of the file to save the embeddings.
        """
        if self.overwrite_json_files or not os.path.exists(filename):
            with open(filename, mode="w", encoding="utf-8") as fp:
                json.dump(
                    [
                        {
                            "text": fel.line,
                            "meta": fel.fel.model_dump(),
                        }
                        for fel in fel_lines
                    ],
                    fp,
                    ensure_ascii=False,
                    indent=2,
                )

    def dump(self, filename: str) -> None:
        """Create dump the embeddings

        :param filename: The base filename to use for the output files. If None, the default path is used.
        """
        base, _ = os.path.splitext(filename)
        self._dump(self._fel0, f"{base}0.json")
        self._dump(self._fel1, f"{base}1.json")
        self._dump(self._fel2, f"{base}2.json")
        ## Make sure to remove the old {base}.npz file
        if os.path.exists(f"{base}.npz"):
            os.remove(f"{base}.npz")
        np.savez(
            base,
            emb0=self._emb0,
            emb1=self._emb1,
            emb2=self._emb2,
        )
