"""
VectorDB — a vector database built from scratch in Python, with HNSW,
KD-Tree, and Brute Force search side-by-side, plus a RAG pipeline over a
local LLM via Ollama.

Author: Parth Mishra
"""

from .types import VectorItem
from .distance import euclidean, cosine, manhattan, get_dist_fn
from .brute_force import BruteForce
from .kdtree import KDTree
from .hnsw import HNSW
from .vector_db import VectorDB, Hit, SearchResult, BenchmarkResult
from .document_db import DocumentDB, DocItem
from .ollama_client import OllamaClient
from .text_chunker import chunk_text
from .demo_data import load_demo

__all__ = [
    "VectorItem",
    "euclidean", "cosine", "manhattan", "get_dist_fn",
    "BruteForce", "KDTree", "HNSW",
    "VectorDB", "Hit", "SearchResult", "BenchmarkResult",
    "DocumentDB", "DocItem",
    "OllamaClient",
    "chunk_text",
    "load_demo",
]
