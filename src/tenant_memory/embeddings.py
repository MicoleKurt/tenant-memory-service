"""Embedding providers behind one small protocol.

* ``HashingEmbedder`` is deterministic and needs no network or key. It is what the
  test suite and local development use. It captures lexical overlap, not meaning.
* ``OpenAIEmbedder`` calls the OpenAI embeddings endpoint for real semantic
  vectors. The key is passed in explicitly and never logged or put in an error.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

import httpx

from .errors import EmbeddingError

_WORD = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    name: str
    dimensions: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """Signed feature hashing of word unigrams and bigrams, L2-normalised."""

    name = "hashing"

    def __init__(self, dimensions: int = 256) -> None:
        if dimensions < 16:
            raise ValueError("dimensions must be at least 16")
        self.dimensions = dimensions

    def _vector(self, text: str) -> list[float]:
        words = _WORD.findall(text.lower())
        features = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
        vec = [0.0] * self.dimensions
        for feature in features:
            digest = hashlib.sha1(feature.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vec[index] += 1.0 if digest[4] & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            vec[0] = 1.0  # cosine distance is undefined for the zero vector
            return vec
        return [v / norm for v in vec]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]


class OpenAIEmbedder:
    name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-large",
        dimensions: int = 1024,
        base_url: str = "https://api.openai.com/v1",
        client: httpx.Client | None = None,
        batch_size: int = 96,
    ) -> None:
        if not api_key:
            raise EmbeddingError("an OpenAI API key is required for the openai embedder")
        self.model = model
        self.dimensions = dimensions
        self._url = base_url.rstrip("/") + "/embeddings"
        self._headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self._client = client or httpx.Client(timeout=30.0)
        self._batch = batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch):
            batch = texts[start : start + self._batch]
            body = {"model": self.model, "input": batch, "dimensions": self.dimensions}
            try:
                response = self._client.post(self._url, headers=self._headers, json=body)
            except httpx.HTTPError as exc:
                raise EmbeddingError(f"embedding request failed: {type(exc).__name__}") from None
            if response.status_code != 200:
                raise EmbeddingError(f"embedding provider answered HTTP {response.status_code}")
            data = sorted(response.json().get("data", []), key=lambda item: item["index"])
            if len(data) != len(batch) or any(len(item["embedding"]) != self.dimensions for item in data):
                raise EmbeddingError("embedding provider returned an unexpected number or size of vectors")
            vectors.extend(item["embedding"] for item in data)
        return vectors
