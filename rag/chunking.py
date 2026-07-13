"""Utilities for splitting raw text into overlapping chunks suitable for embedding."""
from typing import List


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split `text` into overlapping chunks of roughly `chunk_size` characters.

    Args:
        text: The raw document text.
        chunk_size: Target number of characters per chunk.
        overlap: Number of characters shared between consecutive chunks, so
            context is not lost at chunk boundaries.

    Returns:
        A list of text chunks, in order.
    """
    text = text.strip()
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_len:
            break
        start = end - overlap

    return chunks
