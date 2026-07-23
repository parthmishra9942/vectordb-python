"""
Distance / similarity metrics between embedding vectors.

Ported directly from main.cpp's euclidean(), cosine(), manhattan(), and
getDistFn(). Behavior is preserved exactly, including the cosine
zero-vector guard (returns 1.0 -- "maximally distant" -- instead of
dividing by zero).
"""

import math
from typing import Callable, Sequence

DistFn = Callable[[Sequence[float], Sequence[float]], float]


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a)
    nb = sum(y * y for y in b)
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return 1.0 - dot / (math.sqrt(na) * math.sqrt(nb))


def manhattan(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b))


def get_dist_fn(metric: str) -> DistFn:
    if metric == "cosine":
        return cosine
    if metric == "manhattan":
        return manhattan
    return euclidean
