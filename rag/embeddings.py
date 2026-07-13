"""Thin wrapper around a local sentence-transformers model for embeddings."""
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

_DEFAULT_MODEL = "all-MiniLM-L6-v2"


class Embedder:
    """Embeds text using a local sentence-transformers model (no external API calls)."""

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        """Return a (len(texts), dim) float32 array of L2-normalized embeddings."""
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.astype("float32")
