"""FAISS-backed vector index for storing and searching document chunk embeddings."""
from dataclasses import dataclass
from typing import List

import faiss
import numpy as np


@dataclass
class RetrievedChunk:
    text: str
    source: str
    score: float


class VectorIndex:
    """A minimal in-memory FAISS index over normalized embeddings.

    Since embeddings are L2-normalized, inner product search is equivalent
    to cosine similarity search.
    """

    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)
        self._chunks: List[str] = []
        self._sources: List[str] = []

    def add(self, embeddings: np.ndarray, chunks: List[str], sources: List[str]) -> None:
        assert embeddings.shape[0] == len(chunks) == len(sources)
        self.index.add(embeddings)
        self._chunks.extend(chunks)
        self._sources.extend(sources)

    def search(self, query_embedding: np.ndarray, top_k: int = 4) -> List[RetrievedChunk]:
        scores, indices = self.index.search(query_embedding, top_k)
        results: List[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append(
                RetrievedChunk(text=self._chunks[idx], source=self._sources[idx], score=float(score))
            )
        return results
