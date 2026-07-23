"""
HTTP server exposing the VectorDB REST API, and serving the frontend.

Ported from the httplib::Server routes in main.cpp's main() function.
Two deliberate Python-idiomatic changes from the original:

1. **Flask instead of raw httplib-equivalent.** Flask is the standard
   choice for this in Python; cpp-httplib was a reasonable choice in C++
   (header-only, no dependencies) but there's no equivalent reason to
   hand-roll HTTP parsing in Python.
2. **`json.dumps`/`request.get_json()` instead of hand-rolled JSON.** The
   original's jS()/jVec()/extractStr()/extractInt() functions manually
   built and parsed JSON strings because cpp-httplib doesn't include a
   JSON library. Python's standard `json` module does this correctly and
   robustly (handles all escaping, nesting, etc.), so this port uses it
   throughout instead of re-implementing string escaping by hand.

All endpoint paths, request/response field names, and status codes match
the original exactly, so the existing index.html frontend works against
this server completely unmodified.
"""

import os

from flask import Flask, request, jsonify, send_from_directory

from vectordb import (
    VectorDB, DocumentDB, OllamaClient, load_demo, chunk_text, get_dist_fn, cosine,
)

DIMS = 16  # demo vector dimensionality (doc embeddings dim is runtime-determined from Ollama)

app = Flask(__name__, static_folder=None)

db = VectorDB(DIMS)
doc_db = DocumentDB()
ollama = OllamaClient()

load_demo(db)


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.route("/<path:_any>", methods=["OPTIONS"])
@app.route("/", methods=["OPTIONS"])
def cors_preflight(_any=None):
    return "", 204


def _parse_vec_param(raw: str) -> list[float]:
    if not raw:
        return []
    out = []
    for part in raw.split(","):
        try:
            out.append(float(part))
        except ValueError:
            pass
    return out


# ── DEMO VECTOR ENDPOINTS ────────────────────────────────────────────────

@app.route("/search", methods=["GET"])
def search():
    q = _parse_vec_param(request.args.get("v", ""))
    if len(q) != DIMS:
        return jsonify({"error": f"need {DIMS}D vector"})

    try:
        k = int(request.args.get("k", 5))
    except ValueError:
        k = 5
    metric = request.args.get("metric") or "cosine"
    algo = request.args.get("algo") or "hnsw"

    out = db.search(q, k, metric, algo)
    return jsonify({
        "results": [
            {"id": h.id, "metadata": h.metadata, "category": h.category,
             "distance": h.distance, "embedding": h.emb}
            for h in out.hits
        ],
        "latencyUs": out.latency_us,
        "algo": out.algo,
        "metric": out.metric,
    })


@app.route("/insert", methods=["POST"])
def insert():
    body = request.get_json(silent=True) or {}
    meta = body.get("metadata", "")
    cat = body.get("category", "")
    emb = body.get("embedding", [])
    if not meta or len(emb) != DIMS:
        return jsonify({"error": "invalid body"})

    item_id = db.insert(meta, cat, emb, get_dist_fn("cosine"))
    return jsonify({"id": item_id})


@app.route("/delete/<int:item_id>", methods=["DELETE"])
def delete_item(item_id: int):
    ok = db.remove(item_id)
    return jsonify({"ok": ok})


@app.route("/items", methods=["GET"])
def items():
    return jsonify([
        {"id": v.id, "metadata": v.metadata, "category": v.category, "embedding": v.emb}
        for v in db.all()
    ])


@app.route("/benchmark", methods=["GET"])
def benchmark():
    q = _parse_vec_param(request.args.get("v", ""))
    if len(q) != DIMS:
        return jsonify({"error": f"need {DIMS}D vector"})

    try:
        k = int(request.args.get("k", 5))
    except ValueError:
        k = 5
    metric = request.args.get("metric") or "cosine"

    b = db.benchmark(q, k, metric)
    return jsonify({
        "bruteforceUs": b.bruteforce_us,
        "kdtreeUs": b.kdtree_us,
        "hnswUs": b.hnsw_us,
        "itemCount": b.item_count,
    })


@app.route("/hnsw-info", methods=["GET"])
def hnsw_info():
    return jsonify(db.hnsw_info())


# ── DOCUMENT + RAG ENDPOINTS ─────────────────────────────────────────────

@app.route("/doc/insert", methods=["POST"])
def doc_insert():
    body = request.get_json(silent=True) or {}
    title = body.get("title", "")
    text = body.get("text", "")
    if not title or not text:
        return jsonify({"error": "need title and text"})

    chunks = chunk_text(text, 250, 30)
    ids = []

    for i, chunk in enumerate(chunks):
        emb = ollama.embed(chunk)
        if not emb:
            return jsonify({
                "error": "Ollama unavailable. Install from https://ollama.com then run: "
                         "ollama pull nomic-embed-text && ollama pull llama3.2"
            })
        chunk_title = f"{title} [{i + 1}/{len(chunks)}]" if len(chunks) > 1 else title
        ids.append(doc_db.insert(chunk_title, chunk, emb))

    return jsonify({"ids": ids, "chunks": len(chunks), "dims": doc_db.get_dims()})


@app.route("/doc/delete/<int:item_id>", methods=["DELETE"])
def doc_delete(item_id: int):
    ok = doc_db.remove(item_id)
    return jsonify({"ok": ok})


@app.route("/doc/list", methods=["GET"])
def doc_list():
    out = []
    for d in doc_db.all():
        preview = d.text[:120] + ("…" if len(d.text) > 120 else "")
        out.append({
            "id": d.id, "title": d.title, "preview": preview,
            "words": d.text.count(" ") + 1,
        })
    return jsonify(out)


@app.route("/doc/search", methods=["POST"])
def doc_search():
    body = request.get_json(silent=True) or {}
    question = body.get("question", "")
    k = int(body.get("k", 3))
    if not question:
        return jsonify({"error": "need question"})

    q_emb = ollama.embed(question)
    if not q_emb:
        return jsonify({"error": "Ollama unavailable"})

    hits = doc_db.search(q_emb, k)
    return jsonify({
        "contexts": [{"id": d.id, "title": d.title, "distance": dist} for dist, d in hits]
    })


@app.route("/doc/ask", methods=["POST"])
def doc_ask():
    body = request.get_json(silent=True) or {}
    question = body.get("question", "")
    k = int(body.get("k", 3))
    if not question:
        return jsonify({"error": "need question"})

    # Step 1: embed the question
    q_emb = ollama.embed(question)
    if not q_emb:
        return jsonify({"error": "Ollama unavailable"})

    # Step 2: retrieve top-k relevant chunks
    hits = doc_db.search(q_emb, k)

    # Step 3: build prompt
    ctx = "".join(f"[{i + 1}] {d.title}:\n{d.text}\n\n" for i, (_, d) in enumerate(hits))
    prompt = (
        "You are a helpful assistant. Answer the user's question directly. "
        "Use the provided context if it contains relevant information. "
        "If it doesn't, just use your own general knowledge. "
        "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like "
        "'the context doesn't mention'. Just answer the question naturally.\n\n"
        f"Context:\n{ctx}Question: {question}\n\nAnswer:"
    )

    # Step 4: generate answer
    answer = ollama.generate(prompt)

    # Step 5: return everything
    return jsonify({
        "answer": answer,
        "model": ollama.gen_model,
        "contexts": [
            {"id": d.id, "title": d.title, "text": d.text, "distance": dist}
            for dist, d in hits
        ],
        "docCount": doc_db.size(),
    })


@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "ollamaAvailable": ollama.is_available(),
        "embedModel": ollama.embed_model,
        "genModel": ollama.gen_model,
        "docCount": doc_db.size(),
        "docDims": doc_db.get_dims(),
        "demoDims": DIMS,
        "demoCount": db.size(),
    })


@app.route("/stats", methods=["GET"])
def stats():
    return jsonify({
        "count": db.size(),
        "dims": DIMS,
        "algorithms": ["bruteforce", "kdtree", "hnsw"],
        "metrics": ["euclidean", "cosine", "manhattan"],
    })


# ── FRONTEND ──────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return send_from_directory(os.path.dirname(__file__), "index.html")


if __name__ == "__main__":
    ollama_up = ollama.is_available()
    print("=== VectorDB Engine (Python) ===")
    print("http://localhost:8080")
    print(f"{db.size()} demo vectors | {DIMS} dims | HNSW+KD-Tree+BruteForce")
    print(f"Ollama: {'ONLINE' if ollama_up else 'OFFLINE (install from ollama.com)'}")
    if ollama_up:
        print(f"  embed model: {ollama.embed_model}  gen model: {ollama.gen_model}")

    app.run(host="0.0.0.0", port=8080, threaded=True)
