"""CLI demo: ingest sample documents and answer a question grounded in them.

Usage:
    python main.py "What does this project demonstrate?"
"""
import glob
import sys

from rag.pipeline import RAGPipeline

SAMPLE_DOCS_GLOB = "data/sample_docs/*.txt"


def main() -> None:
    question = " ".join(sys.argv[1:]) or "What does this project demonstrate?"

    doc_paths = sorted(glob.glob(SAMPLE_DOCS_GLOB))
    if not doc_paths:
        raise SystemExit(f"No sample documents found at {SAMPLE_DOCS_GLOB}")

    pipeline = RAGPipeline()
    num_chunks = pipeline.ingest(doc_paths)
    print(f"Indexed {num_chunks} chunks from {len(doc_paths)} documents.\n")

    print(pipeline.answer(question))


if __name__ == "__main__":
    main()
