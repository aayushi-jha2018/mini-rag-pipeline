# mini-rag-pipeline

[![CI](https://github.com/aayushi-jha2018/mini-rag-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/aayushi-jha2018/mini-rag-pipeline/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


A small, self-contained **Retrieval-Augmented Generation (RAG)** pipeline demo. It chunks text documents, embeds them locally with `sentence-transformers`, indexes them in FAISS, and retrieves the most relevant passages for a question — the same core pattern used in the production RAG platform described in my [portfolio](https://github.com/aayushi-jha2018/portfolio).

This is intentionally minimal and dependency-light so it runs anywhere without cloud credentials. In production (see my portfolio), the embedding and generation steps are swapped for AWS Bedrock, and retrieval runs against OpenSearch instead of an in-memory FAISS index.

## Architecture

```
data/sample_docs/*.txt
        │
        ▼
  chunk_text()          — rag/chunking.py
        │
        ▼
  Embedder.embed()       — rag/embeddings.py  (sentence-transformers, local)
        │
        ▼
  VectorIndex            — rag/retriever.py   (FAISS, cosine similarity)
        │
        ▼
  RAGPipeline.answer()   — rag/pipeline.py    (retrieval + grounded context)
```

## Project structure

```
mini-rag-pipeline/
├── data/sample_docs/     # sample .txt documents to ingest
├── rag/
│   ├── chunking.py       # splits raw text into overlapping chunks
│   ├── embeddings.py     # wraps a local sentence-transformers model
│   ├── retriever.py      # FAISS vector index + similarity search
│   └── pipeline.py       # ties ingestion + retrieval together
├── main.py               # CLI entry point
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python main.py "What does this project demonstrate?"
```

This ingests every `.txt` file in `data/sample_docs/`, embeds the chunks, and prints the passages most relevant to your question along with their source file and similarity score.

## Extending this demo

- Swap `Embedder` for an AWS Bedrock or OpenAI embedding call to match a production setup.
- Swap the extractive `answer()` method for a real LLM call that generates a fluent response from the retrieved context.
- Replace the in-memory FAISS index with OpenSearch or another managed vector store for scale and persistence.

## License

MIT — feel free to reuse this as a starting point.
