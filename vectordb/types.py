"""
Core data type shared by all three index structures.

Ported from VectorItem in main.cpp.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class VectorItem:
    id: int
    metadata: str
    category: str
    emb: List[float] = field(default_factory=list)
