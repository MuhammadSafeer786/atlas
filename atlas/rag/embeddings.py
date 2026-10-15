"""Dense embeddings.

CONCEPT: semantic retrieval. An embedding model maps text to a vector so that
texts with similar *meaning* land close together (cosine similarity), even
with no shared words ("car" ~ "automobile").

* `FastEmbedEmbedder` - a real local embedding model (BAAI/bge-small-en-v1.5,
  ONNX, CPU-only). Install with `pip install atlas-agents[embeddings]`.
* `HashingEmbedder`   - dependency-free fallback using feature hashing of word
  and character n-grams. Not semantic, but it keeps the vector pipeline
  runnable everywhere and is a useful teaching baseline.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol

from atlas.rag.bm25 import tokenize


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def normalize(v: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))  # inputs are pre-normalised


class HashingEmbedder:
    name = "hashing-512"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _features(self, text: str) -> list[str]:
        words = tokenize(text)
        grams = [f"w:{w}" for w in words] + [f"b:{a}_{b}" for a, b in zip(words, words[1:], strict=False)]
        for w in words:
            padded = f"^{w}$"
            grams += [f"c:{padded[i:i + 3]}" for i in range(len(padded) - 2)]
        return grams

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            v = [0.0] * self.dim
            for g in self._features(text):
                h = int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "little")
                v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
            out.append(normalize(v))
        return out


class FastEmbedEmbedder:
    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding  # optional dependency

        self.name = f"fastembed:{model}"
        self._model = TextEmbedding(model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [normalize([float(x) for x in vec]) for vec in self._model.embed(texts)]


def default_embedder() -> Embedder:
    try:
        return FastEmbedEmbedder()
    except Exception:  # noqa: BLE001 - not installed or model download unavailable
        return HashingEmbedder()
