"""End-to-end RAG pipeline: ingest documents, then answer questions grounded in them."""
import os
from typing import List, Optional

from .chunking import chunk_text
from .embeddings import Embedder
from .retriever import RetrievedChunk, VectorIndex


class RAGPipeline:
    """Ties chunking, embedding, and retrieval together into a single, simple API."""

    def __init__(self, chunk_size: int = 500, overlap: int = 50, top_k: int = 4):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k
        self.embedder = Embedder()
        self.index: Optional[VectorIndex] = None

    def ingest(self, doc_paths: List[str]) -> int:
        """Chunk and embed every document in ``doc_paths``.

        Returns:
            The number of chunks indexed.
        """
        all_chunks: List[str] = []
        all_sources: List[str] = []

        for path in doc_paths:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            chunks = chunk_text(text, self.chunk_size, self.overlap)
            all_chunks.extend(chunks)
            all_sources.extend([os.path.basename(path)] * len(chunks))

        if not all_chunks:
            raise ValueError("No text found in the provided documents.")

        embeddings = self.embedder.embed(all_chunks)
        self.index = VectorIndex(dim=embeddings.shape[1])
        self.index.add(embeddings, all_chunks, all_sources)
        return len(all_chunks)

    def retrieve(self, question: str) -> List[RetrievedChunk]:
        if self.index is None:
            raise RuntimeError("Call ingest() before retrieve().")
        query_embedding = self.embedder.embed([question])
        return self.index.search(query_embedding, top_k=self.top_k)

    def answer(self, question: str) -> str:
        """Return a grounded answer built from the most relevant retrieved chunks.

        This is a simple extractive baseline: it surfaces the most relevant
        passages with their sources. Swap in a call to your LLM of choice
        here (e.g. AWS Bedrock, OpenAI) to generate a fluent answer from the
        same retrieved context.
        """
        results = self.retrieve(question)
        if not results:
            return "No relevant context found."

        lines = [f"Question: {question}", "", "Most relevant passages:"]
        for i, r in enumerate(results, start=1):
            lines.append(f"\n[{i}] (source: {r.source}, score: {r.score:.3f})\n{r.text}")
        return "\n".join(lines)
