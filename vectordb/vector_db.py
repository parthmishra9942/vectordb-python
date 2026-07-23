"""
VectorDB: the unified interface over BruteForce, KDTree, and HNSW —
inserts go into all three (so they can be benchmarked head-to-head),
searches pick one algorithm by name.

Ported from the VectorDB class in main.cpp. The original used
std::lock_guard<std::mutex> around every method since httplib serves
requests on multiple threads; this port uses Python's threading.Lock the
same way, for the same reason (Flask's dev server is also multi-threaded).
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List

from .types import VectorItem
from .distance import DistFn, get_dist_fn
from .brute_force import BruteForce
from .kdtree import KDTree
from .hnsw import HNSW


@dataclass
class Hit:
    id: int
    metadata: str
    category: str
    emb: List[float]
    distance: float


@dataclass
class SearchResult:
    hits: List[Hit]
    latency_us: int
    algo: str
    metric: str


@dataclass
class BenchmarkResult:
    bruteforce_us: int
    kdtree_us: int
    hnsw_us: int
    item_count: int


class VectorDB:
    def __init__(self, dims: int):
        self.dims = dims
        self._store: Dict[int, VectorItem] = {}
        self._bf = BruteForce()
        self._kdt = KDTree(dims)
        self._hnsw = HNSW(m=16, ef_build=200)
        self._lock = threading.Lock()
        self._next_id = 1

    def insert(self, metadata: str, category: str, emb: List[float], dist: DistFn) -> int:
        with self._lock:
            item = VectorItem(id=self._next_id, metadata=metadata, category=category, emb=emb)
            self._next_id += 1
            self._store[item.id] = item
            self._bf.insert(item)
            self._kdt.insert(item)
            self._hnsw.insert(item, dist)
            return item.id

    def remove(self, item_id: int) -> bool:
        with self._lock:
            if item_id not in self._store:
                return False
            del self._store[item_id]
            self._bf.remove(item_id)
            self._hnsw.remove(item_id)
            self._kdt.rebuild(list(self._store.values()))
            return True

    def search(self, q: List[float], k: int, metric: str, algo: str) -> SearchResult:
        with self._lock:
            dfn = get_dist_fn(metric)
            t0 = time.perf_counter()

            if algo == "bruteforce":
                raw = self._bf.knn(q, k, dfn)
            elif algo == "kdtree":
                raw = self._kdt.knn(q, k, dfn)
            else:
                raw = self._hnsw.knn(q, k, 50, dfn)

            latency_us = int((time.perf_counter() - t0) * 1_000_000)

            hits = [
                Hit(id=item_id, metadata=self._store[item_id].metadata,
                    category=self._store[item_id].category, emb=self._store[item_id].emb,
                    distance=d)
                for d, item_id in raw if item_id in self._store
            ]
            return SearchResult(hits=hits, latency_us=latency_us, algo=algo, metric=metric)

    def benchmark(self, q: List[float], k: int, metric: str) -> BenchmarkResult:
        with self._lock:
            dfn = get_dist_fn(metric)

            def timed(fn) -> int:
                t0 = time.perf_counter()
                fn()
                return int((time.perf_counter() - t0) * 1_000_000)

            bf_us = timed(lambda: self._bf.knn(q, k, dfn))
            kd_us = timed(lambda: self._kdt.knn(q, k, dfn))
            hnsw_us = timed(lambda: self._hnsw.knn(q, k, 50, dfn))

            return BenchmarkResult(bruteforce_us=bf_us, kdtree_us=kd_us, hnsw_us=hnsw_us,
                                    item_count=len(self._store))

    def all(self) -> List[VectorItem]:
        with self._lock:
            return list(self._store.values())

    def hnsw_info(self) -> dict:
        with self._lock:
            return self._hnsw.get_info()

    def size(self) -> int:
        with self._lock:
            return len(self._store)
