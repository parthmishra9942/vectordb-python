"""
KD-Tree: binary space partitioning over the embedding dimensions, cycling
through axes at each depth. Prunes subtrees that can't possibly contain a
closer point than the current k-th best ("ball within hyperslab" check).

Ported from KDNode/KDTree in main.cpp. The original used a max-heap
(std::priority_queue, which is max-heap by default in C++) to track the
k best candidates seen so far, popping the worst when it grows past k.
Python's heapq is a min-heap, so we simulate the same max-heap behavior
by pushing (-distance, id) tuples -- this preserves the original's exact
pruning logic (heap top = current worst of the k-best).

Note: unlike the original's manual new/delete + destroy() recursion, this
port relies on Python's garbage collector, so there's no destroy() method
here -- there's nothing to leak.
"""

import heapq
from typing import List, Optional, Tuple

from .types import VectorItem
from .distance import DistFn


class _KDNode:
    __slots__ = ("item", "left", "right")

    def __init__(self, item: VectorItem):
        self.item = item
        self.left: Optional["_KDNode"] = None
        self.right: Optional["_KDNode"] = None


class KDTree:
    def __init__(self, dims: int):
        self.dims = dims
        self.root: Optional[_KDNode] = None

    def insert(self, item: VectorItem):
        self.root = self._ins(self.root, item, 0)

    def _ins(self, node: Optional[_KDNode], item: VectorItem, depth: int) -> _KDNode:
        if node is None:
            return _KDNode(item)
        axis = depth % self.dims
        if item.emb[axis] < node.item.emb[axis]:
            node.left = self._ins(node.left, item, depth + 1)
        else:
            node.right = self._ins(node.right, item, depth + 1)
        return node

    def knn(self, q: List[float], k: int, dist: DistFn) -> List[Tuple[float, int]]:
        # Max-heap of (-distance, id), simulating the original's
        # std::priority_queue<pair<float,int>> (max at top).
        heap: List[Tuple[float, int]] = []
        self._knn(self.root, q, k, 0, dist, heap)
        results = [(-neg_d, i) for neg_d, i in heap]
        results.sort(key=lambda r: r[0])
        return results

    def _knn(self, node: Optional[_KDNode], q: List[float], k: int, depth: int,
              dist: DistFn, heap: List[Tuple[float, int]]):
        if node is None:
            return

        dn = dist(q, node.item.emb)
        worst = -heap[0][0] if heap else None
        if len(heap) < k or dn < worst:
            heapq.heappush(heap, (-dn, node.item.id))
            if len(heap) > k:
                heapq.heappop(heap)

        axis = depth % self.dims
        diff = q[axis] - node.item.emb[axis]
        closer, farther = (node.left, node.right) if diff < 0 else (node.right, node.left)

        self._knn(closer, q, k, depth + 1, dist, heap)

        worst = -heap[0][0] if heap else None
        if len(heap) < k or abs(diff) < worst:
            self._knn(farther, q, k, depth + 1, dist, heap)

    def rebuild(self, items: List[VectorItem]):
        self.root = None
        for item in items:
            self.insert(item)
