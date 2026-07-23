"""
Thin client for the local Ollama REST API — wraps the /api/embeddings and
/api/generate endpoints used for the RAG pipeline.

Ported from the OllamaClient class in main.cpp. The original hand-built
JSON request bodies as raw strings and hand-parsed the "embedding"/
"response" fields out of the JSON response with manual bracket-matching
(parseEmbedding/parseResponse) — necessary in C++ with only httplib
available, but unnecessary in Python. This port uses `requests` for HTTP
and the standard `json` module for encoding/decoding, which is the
idiomatic and more robust Python equivalent of the same logic.

Install: https://ollama.com
Models:  ollama pull nomic-embed-text
         ollama pull llama3.2
"""

from typing import List, Optional

import requests


class OllamaClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 11434):
        self.base_url = f"http://{host}:{port}"
        self.embed_model = "nomic-embed-text"
        self.gen_model = "llama3.2"

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def embed(self, text: str) -> List[float]:
        """Returns an empty list if Ollama is unreachable or the model
        isn't pulled -- matches the original's empty-vector-on-failure
        contract so callers don't need special-case error handling."""
        try:
            r = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.embed_model, "prompt": text},
                timeout=30,
            )
            if r.status_code != 200:
                return []
            return r.json().get("embedding", [])
        except requests.RequestException:
            return []

    def generate(self, prompt: str) -> str:
        try:
            r = requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.gen_model, "prompt": prompt, "stream": False},
                timeout=180,  # LLMs can be slow on CPU
            )
            if r.status_code != 200:
                return "ERROR: Ollama unavailable. Run: ollama serve"
            return r.json().get("response", "")
        except requests.RequestException:
            return "ERROR: Ollama unavailable. Run: ollama serve"
