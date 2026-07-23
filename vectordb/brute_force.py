"""
Brute-force exact k-NN: O(N*d) -- compares the query against every stored
vector. The baseline every other index is measured against.

Ported from the BruteForce class in main.cpp.
"""

from typing import List, Tuple

from .types import VectorItem
from .distance import DistFn


class BruteForce:
    def __init__(self):
        self.items: List[VectorItem] = []

    def insert(self, item: VectorItem):
        self.items.append(item)

    def knn(self, q: List[float], k: int, dist: DistFn) -> List[Tuple[float, int]]:
        results = [(dist(q, v.emb), v.id) for v in self.items]
        results.sort(key=lambda r: r[0])
        return results[:k]

    def remove(self, item_id: int):
        self.items = [v for v in self.items if v.id != item_id]
