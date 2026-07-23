"""
Unit tests for the VectorDB port. Run with: pytest
"""

import sys
import os
import math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vectordb import (
    VectorItem, euclidean, cosine, manhattan, get_dist_fn,
    BruteForce, KDTree, HNSW, VectorDB, DocumentDB, chunk_text, load_demo,
)


# ---------- distance metrics ----------

def test_euclidean_identical_vectors_is_zero():
    assert euclidean([1, 2, 3], [1, 2, 3]) == 0.0


def test_cosine_identical_vectors_is_zero():
    assert abs(cosine([1, 2, 3], [1, 2, 3])) < 1e-9


def test_cosine_orthogonal_vectors_is_one():
    assert abs(cosine([1, 0], [0, 1]) - 1.0) < 1e-9


def test_cosine_zero_vector_guard():
    # Matches the original's guard: near-zero-magnitude vectors return 1.0
    # (maximally distant) instead of raising a divide-by-zero error.
    assert cosine([0, 0, 0], [1, 2, 3]) == 1.0


def test_manhattan():
    assert manhattan([0, 0], [3, 4]) == 7


def test_get_dist_fn_defaults_to_euclidean():
    assert get_dist_fn("unknown") is euclidean
    assert get_dist_fn("cosine") is cosine
    assert get_dist_fn("manhattan") is manhattan


# ---------- brute force ----------

def test_brute_force_knn_returns_closest_first():
    bf = BruteForce()
    bf.insert(VectorItem(1, "a", "cat", [0, 0]))
    bf.insert(VectorItem(2, "b", "cat", [10, 10]))
    bf.insert(VectorItem(3, "c", "cat", [1, 1]))
    results = bf.knn([0, 0], k=2, dist=euclidean)
    assert [r[1] for r in results] == [1, 3]  # id 1 (dist 0), then id 3


def test_brute_force_remove():
    bf = BruteForce()
    bf.insert(VectorItem(1, "a", "cat", [0, 0]))
    bf.remove(1)
    assert bf.knn([0, 0], k=5, dist=euclidean) == []


# ---------- kd-tree ----------

def test_kdtree_matches_brute_force_on_small_set():
    """The core correctness check: KD-Tree's pruned search should return
    the exact same nearest neighbors as brute force for the same data."""
    items = [
        VectorItem(1, "a", "x", [0, 0]),
        VectorItem(2, "b", "x", [5, 5]),
        VectorItem(3, "c", "x", [1, 1]),
        VectorItem(4, "d", "x", [9, 2]),
        VectorItem(5, "e", "x", [2, 8]),
    ]
    bf = BruteForce()
    kdt = KDTree(dims=2)
    for it in items:
        bf.insert(it)
        kdt.insert(it)

    q = [1.5, 1.5]
    bf_result = sorted(bf.knn(q, k=3, dist=euclidean))
    kdt_result = sorted(kdt.knn(q, k=3, dist=euclidean))
    assert bf_result == kdt_result


# ---------- HNSW ----------

def test_hnsw_finds_exact_nearest_on_small_set():
    """HNSW is approximate in general, but on a small, low-dimensional set
    with ef=50 it should reliably find the true nearest neighbor."""
    items = [
        VectorItem(1, "a", "x", [0, 0]),
        VectorItem(2, "b", "x", [10, 10]),
        VectorItem(3, "c", "x", [1, 1]),
        VectorItem(4, "d", "x", [20, 3]),
    ]
    hnsw = HNSW(m=16, ef_build=200)
    for it in items:
        hnsw.insert(it, euclidean)

    result = hnsw.knn([0, 0], k=1, ef=50, dist=euclidean)
    assert result[0][1] == 1  # nearest to [0,0] is item id 1, at distance 0


def test_hnsw_remove():
    hnsw = HNSW(m=16, ef_build=200)
    hnsw.insert(VectorItem(1, "a", "x", [0, 0]), euclidean)
    hnsw.insert(VectorItem(2, "b", "x", [1, 1]), euclidean)
    hnsw.remove(1)
    assert len(hnsw) == 1
    result = hnsw.knn([0, 0], k=5, ef=50, dist=euclidean)
    assert all(item_id != 1 for _, item_id in result)


# ---------- VectorDB integration ----------

def test_vector_db_insert_and_search_all_algos_agree():
    """All three algorithms should return the same top-1 result on simple,
    well-separated data -- this is the core 'compare all algos' feature."""
    db = VectorDB(dims=4)
    dist = get_dist_fn("cosine")
    db.insert("near", "cat", [1, 0, 0, 0], dist)
    db.insert("far", "cat", [0, 0, 0, 1], dist)

    q = [0.9, 0.1, 0, 0]
    for algo in ("bruteforce", "kdtree", "hnsw"):
        result = db.search(q, k=1, metric="cosine", algo=algo)
        assert result.hits[0].metadata == "near", f"{algo} disagreed"


def test_vector_db_remove():
    db = VectorDB(dims=2)
    dist = get_dist_fn("euclidean")
    item_id = db.insert("x", "cat", [1, 1], dist)
    assert db.size() == 1
    assert db.remove(item_id) is True
    assert db.size() == 0
    assert db.remove(item_id) is False  # already gone


def test_load_demo_inserts_twenty_items():
    db = VectorDB(dims=16)
    load_demo(db)
    assert db.size() == 20


# ---------- text chunker ----------

def test_chunk_text_short_text_single_chunk():
    text = "just a few words here"
    assert chunk_text(text, chunk_words=250, overlap_words=30) == [text]


def test_chunk_text_long_text_overlaps():
    words = [f"word{i}" for i in range(600)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_words=250, overlap_words=30)
    assert len(chunks) > 1
    # Consecutive chunks should overlap by the configured word count.
    first_chunk_words = chunks[0].split()
    second_chunk_words = chunks[1].split()
    assert first_chunk_words[-30:] == second_chunk_words[:30]


def test_chunk_text_empty():
    assert chunk_text("") == []


# ---------- DocumentDB ----------

def test_document_db_search_respects_max_dist():
    doc_db = DocumentDB()
    doc_db.insert("Doc A", "some text about cats", [1.0, 0.0, 0.0])
    doc_db.insert("Doc B", "unrelated text about rockets", [-1.0, 0.0, 0.0])

    # Querying with the exact "Doc A" embedding should return Doc A within
    # max_dist, and should not return the maximally-dissimilar Doc B.
    results = doc_db.search([1.0, 0.0, 0.0], k=2, max_dist=0.7)
    titles = [d.title for _, d in results]
    assert "Doc A" in titles
    assert "Doc B" not in titles
