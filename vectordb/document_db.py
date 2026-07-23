"""
DocumentDB: HNSW index over real Ollama embeddings (768D for
nomic-embed-text), as opposed to VectorDB's fixed 16D demo vectors. This
is what the RAG pipeline searches over.

Ported from the DocumentDB class in main.cpp.
"""

import threading
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .types import VectorItem
from .distance import cosine
from .hnsw import HNSW
from .brute_force import BruteForce


@dataclass
class DocItem:
    id: int
    title: str
    text: str
    emb: List[float]


class DocumentDB:
    def __init__(self):
        self._hnsw = HNSW(m=16, ef_build=200)
        self._bf = BruteForce()  # brute-force fallback for small sets
        self._store: Dict[int, DocItem] = {}
        self._lock = threading.Lock()
        self._next_id = 1
        self._dims = 0  # determined from the first inserted embedding

    def insert(self, title: str, text: str, emb: List[float]) -> int:
        with self._lock:
            if self._dims == 0:
                self._dims = len(emb)
            item = DocItem(id=self._next_id, title=title, text=text, emb=emb)
            self._next_id += 1
            self._store[item.id] = item

            vi = VectorItem(id=item.id, metadata=title, category="doc", emb=emb)
            self._hnsw.insert(vi, cosine)
            self._bf.insert(vi)
            return item.id

    def search(self, q: List[float], k: int, max_dist: float = 0.7) -> List[Tuple[float, DocItem]]:
        """Top-k most similar chunks. Uses brute force for small stores
        (HNSW's approximate search isn't worth it below ~10 items) and
        HNSW once the store grows -- same threshold as the original."""
        with self._lock:
            if not self._store:
                return []
            if len(self._store) < 10:
                raw = self._bf.knn(q, k, cosine)
            else:
                raw = self._hnsw.knn(q, k, 50, cosine)

            return [(d, self._store[item_id]) for d, item_id in raw
                    if item_id in self._store and d <= max_dist]

    def remove(self, item_id: int) -> bool:
        with self._lock:
            if item_id not in self._store:
                return False
            del self._store[item_id]
            self._hnsw.remove(item_id)
            self._bf.remove(item_id)
            return True

    def all(self) -> List[DocItem]:
        with self._lock:
            return list(self._store.values())

    def size(self) -> int:
        with self._lock:
            return len(self._store)

    def get_dims(self) -> int:
        return self._dims
