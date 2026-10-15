"""Knowledge base: SQLite storage + hybrid retrieval.

CONCEPT: Retrieval-Augmented Generation (RAG). Instead of hoping the model
memorised your private documents, retrieve the relevant passages at question
time and hand them to the model with their sources, so answers are grounded
and citable.

Pipeline:  ingest -> chunk -> embed -> store          (offline)
           query  -> BM25 + vector search -> RRF fuse -> top-k   (online)

Reciprocal Rank Fusion merges ranked lists without needing comparable scores:
    RRF(d) = sum over rankers of 1 / (k + rank(d)),  k = 60
A document ranked well by *both* retrievers rises to the top.

SQLite keeps the project dependency-free; the same interface maps directly
onto pgvector, Qdrant or similar at scale.
"""

from __future__ import annotations

import html
import re
import sqlite3
import threading
from array import array
from dataclasses import dataclass
from pathlib import Path

from atlas.rag.bm25 import BM25
from atlas.rag.chunking import chunk_text
from atlas.rag.embeddings import Embedder, cosine, default_embedder

SUPPORTED_SUFFIXES = {".md", ".txt", ".rst", ".py", ".json", ".yaml", ".yml", ".csv", ".html", ".htm"}
RRF_K = 60


@dataclass
class SearchHit:
    chunk_id: int
    source: str
    title: str
    heading: str
    text: str
    score: float


def _strip_html(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw)
    return html.unescape(re.sub(r"<[^>]+>", " ", raw))


class KnowledgeBase:
    def __init__(self, path: Path | str = ":memory:", embedder: Embedder | None = None):
        if isinstance(path, Path):
            path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.embedder = embedder or default_embedder()
        self._lock = threading.Lock()
        self._bm25: BM25 | None = None
        self._rows: list[tuple[int, str, str, str, str, list[float]]] = []
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY, source TEXT UNIQUE, title TEXT,
                added TEXT DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY, doc_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                idx INTEGER, heading TEXT, text TEXT, embedding BLOB);
            """
        )
        self._check_embedder()

    # ------------------------------------------------------------- ingestion
    def _check_embedder(self) -> None:
        row = self.db.execute("SELECT value FROM meta WHERE key='embedder'").fetchone()
        if row and row[0] != self.embedder.name:
            self._reembed_all()  # vectors from different models are not comparable
        self.db.execute("INSERT OR REPLACE INTO meta VALUES ('embedder', ?)", (self.embedder.name,))
        self.db.commit()

    def _reembed_all(self) -> None:
        rows = self.db.execute("SELECT id, heading, text FROM chunks").fetchall()
        vectors = self.embedder.embed([f"{h}\n{t}" for _, h, t in rows]) if rows else []
        for (cid, _, _), vec in zip(rows, vectors, strict=True):
            self.db.execute("UPDATE chunks SET embedding=? WHERE id=?", (array("f", vec).tobytes(), cid))

    def add_text(self, text: str, source: str, title: str | None = None) -> int:
        chunks = chunk_text(text)
        if not chunks:
            return 0
        vectors = self.embedder.embed([f"{c.heading}\n{c.text}" for c in chunks])
        with self._lock:
            self.db.execute("DELETE FROM documents WHERE source=?", (source,))  # re-ingest = replace
            self.db.execute("DELETE FROM chunks WHERE doc_id NOT IN (SELECT id FROM documents)")
            cur = self.db.execute("INSERT INTO documents (source, title) VALUES (?, ?)",
                                  (source, title or Path(source).stem))
            doc_id = cur.lastrowid
            self.db.executemany(
                "INSERT INTO chunks (doc_id, idx, heading, text, embedding) VALUES (?, ?, ?, ?, ?)",
                [(doc_id, c.index, c.heading, c.text, array("f", v).tobytes())
                 for c, v in zip(chunks, vectors, strict=True)],
            )
            self.db.commit()
            self._bm25 = None  # invalidate in-memory index
        return len(chunks)

    def add_path(self, path: Path) -> dict[str, int]:
        files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
        added: dict[str, int] = {}
        for f in files:
            if f.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            raw = f.read_text(encoding="utf-8", errors="replace")
            text = _strip_html(raw) if f.suffix.lower() in {".html", ".htm"} else raw
            added[f.as_posix()] = self.add_text(text, source=f.as_posix(), title=f.stem)
        return added

    # ------------------------------------------------------------- retrieval
    def _load(self) -> None:
        if self._bm25 is not None:
            return
        rows = self.db.execute(
            "SELECT c.id, d.source, d.title, c.heading, c.text, c.embedding "
            "FROM chunks c JOIN documents d ON d.id = c.doc_id ORDER BY c.id"
        ).fetchall()
        self._rows = [(cid, src, title, head, text, list(array("f", emb))) for cid, src, title, head, text, emb in rows]
        self._bm25 = BM25([f"{r[2]} {r[3]} {r[4]}" for r in self._rows])

    def search(self, query: str, k: int = 5, mode: str = "hybrid") -> list[SearchHit]:
        with self._lock:
            self._load()
            rows, bm25 = self._rows, self._bm25
        if not rows or bm25 is None:
            return []
        candidates = max(k * 4, 20)
        rankings: list[list[int]] = []
        if mode in ("hybrid", "keyword"):
            lexical = bm25.scores(query)
            rankings.append([i for i in sorted(range(len(rows)), key=lambda i: -lexical[i])
                             if lexical[i] > 0][:candidates])
        if mode in ("hybrid", "vector"):
            qv = self.embedder.embed([query])[0]
            dense = [cosine(qv, r[5]) for r in rows]
            rankings.append(sorted(range(len(rows)), key=lambda i: -dense[i])[:candidates])

        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, i in enumerate(ranking):
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
        best = sorted(fused, key=lambda i: -fused[i])[:k]
        return [SearchHit(rows[i][0], rows[i][1], rows[i][2], rows[i][3], rows[i][4], round(fused[i], 5))
                for i in best]

    def stats(self) -> dict[str, int | str]:
        docs = self.db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunks = self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return {"documents": docs, "chunks": chunks, "embedder": self.embedder.name}

    def sources(self) -> list[str]:
        return [r[0] for r in self.db.execute("SELECT source FROM documents ORDER BY source")]
