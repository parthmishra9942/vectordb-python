"""
HNSW — Hierarchical Navigable Small World graph index.

This is the same algorithm used by production vector databases like
Pinecone, Weaviate, Chroma, and Milvus: a multilayer graph where each
layer is progressively sparser. Search starts at the top (sparse) layer
and greedily descends, "zooming in" until it reaches layer 0, giving
roughly O(log N) search instead of brute force's O(N).

Ported from the HNSW class in main.cpp. C++'s std::priority_queue is
max-heap by default; Python's heapq is min-heap-only, so wherever the
original used a max-heap (the `found` set of best candidates, which needs
fast access to the *worst* of the current best-k), this port pushes
negated distances to get the same behavior. The `cands` frontier, which
the original built with std::greater<> (making it a min-heap), maps
directly onto heapq with no negation needed.

The random level assignment is also ported exactly, including seeding
the RNG with 42 (matching `rng(42)` in the constructor) — this is
deliberate in the original so graph structure is reproducible run to
run, and is preserved here for the same reason.
"""

import heapq
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .types import VectorItem
from .distance import DistFn


@dataclass
class _Node:
    item: VectorItem
    max_layer: int
    nbrs: List[List[int]]  # nbrs[layer] = list of neighbor ids at that layer


class HNSW:
    def __init__(self, m: int = 16, ef_build: int = 200):
        self.M = m
        self.M0 = 2 * m
        self.ef_build = ef_build
        self.mL = 1.0 / math.log(float(m))
        self.top_layer = -1
        self.entry_pt = -1
        self._rng = random.Random(42)  # seeded to match the original's rng(42)
        self.G: Dict[int, _Node] = {}

    def __len__(self) -> int:
        return len(self.G)

    def _rand_level(self) -> int:
        u = self._rng.random()
        # Guard against log(0) -- Python's random() can return 0.0, C++'s
        # uniform_real_distribution effectively never does in practice,
        # but we guard here for robustness rather than silently matching
        # undefined behavior.
        u = max(u, 1e-12)
        return int(math.floor(-math.log(u) * self.mL))

    def _search_layer(self, q: List[float], ep: int, ef: int, layer: int,
                       dist: DistFn) -> List[Tuple[float, int]]:
        visited = {ep}
        d0 = dist(q, self.G[ep].item.emb)

        # Candidate frontier: min-heap on distance (closest first to explore),
        # matches the original's std::greater<>-ordered priority_queue.
        cands: List[Tuple[float, int]] = [(d0, ep)]

        # Best-found-so-far: max-heap (worst of the best-k at the top) so we
        # can cheaply check/evict the worst candidate -- stored as negated
        # distances since heapq is min-heap only.
        found: List[Tuple[float, int]] = [(-d0, ep)]

        while cands:
            cd, cid = heapq.heappop(cands)
            worst_found = -found[0][0]
            if len(found) >= ef and cd > worst_found:
                break

            node = self.G.get(cid)
            if node is None or layer >= len(node.nbrs):
                continue

            for nid in node.nbrs[layer]:
                if nid in visited or nid not in self.G:
                    continue
                visited.add(nid)
                nd = dist(q, self.G[nid].item.emb)
                worst_found = -found[0][0]
                if len(found) < ef or nd < worst_found:
                    heapq.heappush(cands, (nd, nid))
                    heapq.heappush(found, (-nd, nid))
                    if len(found) > ef:
                        heapq.heappop(found)

        result = [(-neg_d, i) for neg_d, i in found]
        result.sort(key=lambda r: r[0])
        return result

    @staticmethod
    def _select_neighbors(cands: List[Tuple[float, int]], max_m: int) -> List[int]:
        # cands is already sorted ascending by _search_layer, so this is
        # just "take the nearest max_m" -- matches selectNbrs() exactly.
        return [cid for _, cid in cands[:max_m]]

    def insert(self, item: VectorItem, dist: DistFn):
        item_id = item.id
        level = self._rand_level()
        self.G[item_id] = _Node(item=item, max_layer=level, nbrs=[[] for _ in range(level + 1)])

        if self.entry_pt == -1:
            self.entry_pt = item_id
            self.top_layer = level
            return

        ep = self.entry_pt

        # Descend from the top layer down to (but not including) this
        # item's own top layer, greedily narrowing the entry point.
        for lc in range(self.top_layer, level, -1):
            if lc < len(self.G[ep].nbrs):
                w = self._search_layer(item.emb, ep, 1, lc, dist)
                if w:
                    ep = w[0][1]

        # From min(topLayer, level) down to 0: find neighbors at this
        # layer, connect bidirectionally, and prune over-full neighbor
        # lists back down to max_m by keeping only the closest.
        for lc in range(min(self.top_layer, level), -1, -1):
            w = self._search_layer(item.emb, ep, self.ef_build, lc, dist)
            max_m = self.M0 if lc == 0 else self.M
            selected = self._select_neighbors(w, max_m)
            self.G[item_id].nbrs[lc] = selected

            for nid in selected:
                if nid not in self.G:
                    continue
                neighbor_node = self.G[nid]
                if len(neighbor_node.nbrs) <= lc:
                    neighbor_node.nbrs.extend([] for _ in range(lc + 1 - len(neighbor_node.nbrs)))
                conn = neighbor_node.nbrs[lc]
                conn.append(item_id)
                if len(conn) > max_m:
                    scored = [(dist(neighbor_node.item.emb, self.G[c].item.emb), c)
                              for c in conn if c in self.G]
                    scored.sort(key=lambda r: r[0])
                    neighbor_node.nbrs[lc] = [c for _, c in scored[:max_m]]

            if w:
                ep = w[0][1]

        if level > self.top_layer:
            self.top_layer = level
            self.entry_pt = item_id

    def knn(self, q: List[float], k: int, ef: int, dist: DistFn) -> List[Tuple[float, int]]:
        if self.entry_pt == -1:
            return []

        ep = self.entry_pt
        for lc in range(self.top_layer, 0, -1):
            if lc < len(self.G[ep].nbrs):
                w = self._search_layer(q, ep, 1, lc, dist)
                if w:
                    ep = w[0][1]

        w = self._search_layer(q, ep, max(ef, k), 0, dist)
        return w[:k]

    def remove(self, item_id: int):
        if item_id not in self.G:
            return
        for node in self.G.values():
            for layer_neighbors in node.nbrs:
                if item_id in layer_neighbors:
                    layer_neighbors.remove(item_id)

        if self.entry_pt == item_id:
            self.entry_pt = -1
            for other_id in self.G:
                if other_id != item_id:
                    self.entry_pt = other_id
                    break

        del self.G[item_id]

    def get_info(self) -> dict:
        """Graph structure snapshot for the visualizer UI (nodes/edges per
        layer). Ported from HNSW::GraphInfo / getInfo()."""
        max_l = max(self.top_layer + 1, 1)
        nodes_per_layer = [0] * max_l
        edges_per_layer = [0] * max_l
        nodes = []
        edges = []

        for node_id, node in self.G.items():
            nodes.append({
                "id": node_id,
                "metadata": node.item.metadata,
                "category": node.item.category,
                "maxLyr": node.max_layer,
            })
            for lc in range(min(node.max_layer, max_l - 1) + 1):
                nodes_per_layer[lc] += 1
                if lc < len(node.nbrs):
                    for nid in node.nbrs[lc]:
                        if node_id < nid:  # count each undirected edge once
                            edges_per_layer[lc] += 1
                            edges.append({"src": node_id, "dst": nid, "lyr": lc})

        return {
            "topLayer": self.top_layer,
            "nodeCount": len(self.G),
            "nodesPerLayer": nodes_per_layer,
            "edgesPerLayer": edges_per_layer,
            "nodes": nodes,
            "edges": edges,
        }
