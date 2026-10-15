"""BM25 lexical retrieval, implemented from scratch.

CONCEPT: sparse (keyword) retrieval. BM25 scores a document by how often the
query terms appear in it, discounted by how common each term is across the
corpus (IDF) and normalised for document length. It excels at exact matches
- names, error codes, IDs - exactly where dense embeddings are weakest.
That complementarity is why production RAG is usually *hybrid*.

    score(D, Q) = sum_t IDF(t) * f(t,D) * (k1 + 1) / (f(t,D) + k1 * (1 - b + b * |D| / avgdl))
"""

from __future__ import annotations

import math
import re
from collections import Counter

STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has", "have",
    "how", "i", "if", "in", "into", "is", "it", "its", "of", "on", "or", "that", "the",
    "their", "them", "then", "there", "these", "they", "this", "to", "was", "were", "what",
    "when", "where", "which", "who", "why", "will", "with", "you", "your",
})
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(d)) for d in documents]
        self.lengths = [sum(c.values()) for c in self.docs]
        self.avgdl = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        df: Counter[str] = Counter()
        for c in self.docs:
            df.update(c.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        out = []
        for counts, length in zip(self.docs, self.lengths, strict=True):
            s = 0.0
            for t in terms:
                f = counts.get(t)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * length / (self.avgdl or 1))
                s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / denom
            out.append(s)
        return out
