"""Document chunking.

CONCEPT: retrieval quality is decided at ingestion time. Chunks must be small
enough to be specific (a whole document matches everything weakly) yet large
enough to stand alone. This splitter is structure-aware:

1. Split Markdown on headings, and remember the heading path ("Guide > Setup")
   so every chunk carries its context - this is "contextual chunking".
2. Pack paragraphs into ~`max_chars` windows.
3. Overlap consecutive windows so a fact on a boundary appears whole in one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


@dataclass
class Chunk:
    text: str
    heading: str
    index: int


def _sections(text: str) -> list[tuple[str, str]]:
    path: list[str] = []
    sections: list[tuple[str, str]] = []
    buf: list[str] = []
    for line in text.splitlines():
        m = HEADING_RE.match(line)
        if m:
            if buf:
                sections.append((" > ".join(path), "\n".join(buf).strip()))
                buf = []
            level = len(m.group(1))
            path = path[: level - 1] + [m.group(2)]
        else:
            buf.append(line)
    if buf:
        sections.append((" > ".join(path), "\n".join(buf).strip()))
    return [(h, body) for h, body in sections if body]


def _split_long(paragraph: str, max_chars: int) -> list[str]:
    if len(paragraph) <= max_chars:
        return [paragraph]
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    out, cur = [], ""
    for s in sentences:
        while len(s) > max_chars:  # pathological: no sentence breaks
            out.append(s[:max_chars])
            s = s[max_chars:]
        if len(cur) + len(s) + 1 > max_chars and cur:
            out.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        out.append(cur)
    return out


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200) -> list[Chunk]:
    chunks: list[Chunk] = []
    for heading, body in _sections(text) or [("", text)]:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        pieces = [piece for p in paragraphs for piece in _split_long(p, max_chars)]
        window = ""
        for piece in pieces:
            if window and len(window) + len(piece) + 2 > max_chars:
                chunks.append(Chunk(window, heading, len(chunks)))
                tail = window[-overlap:] if overlap else ""
                # start the overlap on a word boundary
                tail = tail[tail.find(" ") + 1:] if " " in tail else tail
                window = f"{tail}\n\n{piece}".strip() if tail else piece
            else:
                window = f"{window}\n\n{piece}".strip()
        if window:
            chunks.append(Chunk(window, heading, len(chunks)))
    return chunks
