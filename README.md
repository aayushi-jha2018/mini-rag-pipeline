# mini-rag-pipeline

[![CI](https://github.com/aayushi-jha2018/mini-rag-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/aayushi-jha2018/mini-rag-pipeline/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A small, self-contained retrieval pipeline: chunks text documents, embeds them locally with `sentence-transformers`, indexes them in FAISS, and retrieves the passages most relevant to a question. Same core retrieval pattern as the production RAG platform in my [portfolio](https://github.com/aayushi-jha2018/portfolio), just without the cloud dependencies.

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

This ingests every `.txt` file in `data/sample_docs/`, embeds the chunks, and prints the passages most relevant to your question, along with their source file and similarity score.

## What's actually happening under the hood

```
data/sample_docs/*.txt
  -> chunk_text()          rag/chunking.py        (splits into overlapping chunks)
  -> Embedder.embed()      rag/embeddings.py      (sentence-transformers, local)
  -> VectorIndex           rag/retriever.py       (FAISS, cosine similarity)
  -> RAGPipeline.answer()  rag/pipeline.py        (retrieval + grounded context)
```

## A note on the "generation" part

`answer()` is extractive, not generative -- it returns the retrieved passages themselves rather than an LLM-generated summary of them. That keeps this repo dependency-light and free of API costs, but it means the "G" in RAG is really only implemented on the retrieval side here. In production (see my portfolio), this same retrieval step feeds a real LLM call that generates a grounded answer from the retrieved context; swapping that in here would mean replacing the extractive logic inside `rag/pipeline.py`, not restructuring the retrieval pipeline itself.

## Where this would change for production

- Embeddings: swap the local sentence-transformers model for an AWS Bedrock or OpenAI embedding call.
- Vector store: replace the in-memory FAISS index with OpenSearch (or another managed store) for scale and persistence beyond a single process.
- Answering: replace the extractive `answer()` with a real LLM call, as above.

## Layout

- `data/sample_docs/` -- sample .txt documents to ingest
- `rag/chunking.py` -- splits raw text into overlapping chunks
- `rag/embeddings.py` -- wraps a local sentence-transformers model
- `rag/retriever.py` -- FAISS vector index + similarity search
- `rag/pipeline.py` -- ties ingestion + retrieval together
- `main.py` -- CLI entry point

MIT license.
