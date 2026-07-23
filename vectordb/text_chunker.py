"""
Splits long text into overlapping word-count chunks, so each chunk stays
small enough to embed meaningfully and retrieval can pinpoint a specific
passage rather than an entire document.

Ported from chunkText() in main.cpp.
"""

from typing import List


def chunk_text(text: str, chunk_words: int = 250, overlap_words: int = 30) -> List[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text]

    chunks = []
    step = chunk_words - overlap_words
    i = 0
    while i < len(words):
        end = min(i + chunk_words, len(words))
        chunks.append(" ".join(words[i:end]))
        if end == len(words):
            break
        i += step
    return chunks
