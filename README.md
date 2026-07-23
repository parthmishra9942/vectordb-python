# VectorDB (Python)

**Author:** Parth Mishra ([github.com/parthmishra9942](https://github.com/parthmishra9942))

A vector database built from scratch in Python — implements **HNSW**,
**KD-Tree**, and **Brute Force** search side-by-side, plus a full **RAG
(Retrieval-Augmented Generation) pipeline** powered by a local LLM via
[Ollama](https://ollama.com). Includes a web UI for searching, comparing
algorithm speed, visualizing the HNSW graph, and chatting with your own
documents.

This is a Python port of an original C++ implementation, restructured
into a tested, modular Python package.

> Built as an educational project to show how production vector databases
> like Pinecone, Weaviate, and Chroma actually work under the hood.

---

## What this project does

| Feature | Description |
|---|---|
| **3 Search Algorithms** | HNSW (production-grade), KD-Tree, Brute Force — run all three and compare speed |
| **3 Distance Metrics** | Cosine similarity, Euclidean distance, Manhattan distance |
| **16D Demo Vectors** | 20 pre-loaded semantic vectors across 4 categories (CS, Math, Food, Sports) |
| **2D PCA Scatter Plot** | Live visualization of semantic space — watch clusters form |
| **Real Document Embedding** | Paste any text → Ollama embeds it with `nomic-embed-text` (768D) |
| **RAG Pipeline** | Ask questions about your documents → HNSW retrieves context → local LLM answers |
| **Full REST API** | CRUD endpoints: insert, delete, search, benchmark, hnsw-info |

## How it works

```
Your Text
    │
    ▼
Ollama (nomic-embed-text)          ← converts text to a 768-dimensional vector
    │
    ▼
HNSW Index (Python)                ← indexes the vector in a multilayer graph
    │
    ▼
Semantic Search                    ← finds nearest neighbors in vector space
    │
    ▼
Ollama (llama3.2)                  ← reads retrieved chunks, generates an answer
    │
    ▼
Answer
```

**HNSW (Hierarchical Navigable Small World)** is the same algorithm used
by Pinecone, Weaviate, Chroma, and Milvus. It builds a multilayer graph
where each layer is progressively sparser — searches start at the top
layer and zoom in, achieving roughly O(log N) complexity instead of O(N)
for brute force.

## Project structure

```
vectordb/
  types.py          # VectorItem
  distance.py        # euclidean, cosine, manhattan
  brute_force.py      # Exact O(N) baseline
  kdtree.py           # Binary space partitioning index
  hnsw.py             # The core algorithm — multilayer graph index
  vector_db.py        # Unified interface over all 3 (16D demo vectors)
  document_db.py       # HNSW-only index for real Ollama embeddings (768D)
  ollama_client.py     # HTTP client -> /api/embeddings + /api/generate
  text_chunker.py      # Splits long text into overlapping chunks
  demo_data.py         # The 20 pre-loaded demo vectors
app.py                 # Flask server + REST API (all original endpoints)
index.html              # Frontend (unchanged from the original — same API contract)
tests/test_vectordb.py  # 18 unit tests (pytest)
```

## Running it

Requires Python 3.10+.

```bash
pip install -r requirements.txt

# (Optional, for the Documents/RAG tabs) install Ollama and pull models:
#   https://ollama.com
#   ollama pull nomic-embed-text
#   ollama pull llama3.2

python3 app.py
```

Then open **http://localhost:8080** — the Search tab (demo vectors) and
Compare-Algos benchmark work immediately with no setup. The Documents and
Ask-AI tabs need Ollama running locally.

Run the tests:
```bash
pip install pytest
pytest tests/ -v
```

## What changed in the Python port (and why)

This isn't a line-by-line translation — a few things were done the
Python-idiomatic way instead of mechanically copying the C++ approach:

- **Flask instead of a hand-rolled HTTP server.** The original used
  cpp-httplib (a header-only C++ HTTP library, since C++ has no batteries-
  included web server). Flask is the standard choice for this in Python.
- **`json` module instead of hand-written JSON parsing.** The original
  manually built and parsed JSON strings (`jS()`, `jVec()`, `extractStr()`,
  `extractInt()`) because cpp-httplib doesn't include a JSON library.
  Python's `json` module does this correctly and handles edge cases
  (escaping, nesting, unicode) that the original's string-splicing
  approach didn't need to worry about at this project's scale, but would
  have on more complex inputs.
- **`requests` instead of raw HTTP calls to Ollama.** Same reasoning —
  no reason to hand-build HTTP request bodies in Python.

**Everything algorithmic is a faithful, verified port**: HNSW's layer
assignment, greedy graph descent, and neighbor-pruning logic; KD-Tree's
axis-cycling and subtree-pruning logic; the RAG prompt construction. All
covered by unit tests, including a direct correctness check that HNSW,
KD-Tree, and Brute Force agree on the same nearest-neighbor result.

## Algorithm deep dive

### HNSW (Hierarchical Navigable Small World)

Nodes are inserted into a multilayer graph. Each node randomly gets
assigned a maximum layer (via an exponential distribution — most nodes
only exist at layer 0, a few reach higher layers). Layer 0 has all nodes
with many connections; higher layers have exponentially fewer nodes with
longer-range connections.

**Insert:** Start at the top layer, greedily find the nearest node, drop
a layer, repeat. At each layer from the node's assigned max down to 0,
run a beam search (`ef_construction=200`) and connect to the `M` nearest
neighbors bidirectionally, pruning any neighbor list that grows too large.

**Search:** Same greedy descent from the top layer. At layer 0, expand to
`ef` nearest candidates using a bounded best-first search.

**Why it's fast:** The upper layers act like a highway — you quickly get
to the right neighborhood, then zoom in at layer 0.

### KD-Tree

Binary space partitioning. Each node splits space along one dimension
(cycling through all dimensions with depth). Search prunes entire
subtrees when the closest possible point in that subtree can't beat the
current k-th best distance.

**Weakness:** Degrades in high dimensions (curse of dimensionality) —
works well for ≤20D, becomes close to brute force at 768D, which is why
`DocumentDB` (which stores real 768D Ollama embeddings) uses HNSW rather
than KD-Tree.

## License

MIT — see LICENSE.
