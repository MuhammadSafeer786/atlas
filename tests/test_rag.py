from pathlib import Path

from atlas.evals.runner import run_suite
from atlas.rag.bm25 import BM25, tokenize
from atlas.rag.chunking import chunk_text
from atlas.rag.embeddings import HashingEmbedder, cosine
from atlas.rag.store import KnowledgeBase

ROOT = Path(__file__).resolve().parents[1]


def test_chunking_keeps_heading_paths_and_overlap():
    doc = "# Guide\n\n## Setup\n\n" + "\n\n".join(f"Paragraph {i} " + "word " * 60 for i in range(8))
    chunks = chunk_text(doc, max_chars=600, overlap=100)
    assert len(chunks) > 1
    assert all(c.heading == "Guide > Setup" for c in chunks)
    assert all(len(c.text) <= 600 + 100 for c in chunks)
    # overlap: the tail of one chunk reappears at the head of the next
    assert chunks[0].text[-40:].split()[-1] in chunks[1].text


def test_bm25_prefers_exact_rare_terms():
    docs = ["rate limits and retries", "error NIMBUS_DIM_MISMATCH means wrong dimension", "pricing plans"]
    scores = BM25(docs).scores("NIMBUS_DIM_MISMATCH")
    assert scores.index(max(scores)) == 1
    assert "the" not in tokenize("the rate")


def test_hashing_embedder_similarity():
    e = HashingEmbedder()
    a, b, c = e.embed(["rate limit exceeded", "rate limits exceeded quickly", "chocolate cake recipe"])
    assert cosine(a, b) > cosine(a, c)


def test_kb_hybrid_search_and_reingest(knowledge_dir, tmp_path):
    kb = KnowledgeBase(tmp_path / "kb.db", embedder=HashingEmbedder())
    added = kb.add_path(knowledge_dir)
    assert len(added) == 5 and kb.stats()["documents"] == 5
    hits = kb.search("How much is the Growth plan per month?", k=3)
    assert "pricing.md" in hits[0].source and "$249" in hits[0].text

    # Re-ingesting the same source replaces rather than duplicates
    before = kb.stats()["chunks"]
    kb.add_path(knowledge_dir / "pricing.md")
    assert kb.stats()["chunks"] == before

    # Persistence: a new handle on the same file sees the data
    assert KnowledgeBase(tmp_path / "kb.db", embedder=HashingEmbedder()).stats()["documents"] == 5


def test_embedder_change_triggers_reembed(knowledge_dir, tmp_path):
    class Tiny(HashingEmbedder):
        name = "tiny-16"

        def __init__(self):
            super().__init__(dim=16)

    KnowledgeBase(tmp_path / "kb.db", embedder=HashingEmbedder()).add_path(knowledge_dir / "pricing.md")
    kb = KnowledgeBase(tmp_path / "kb.db", embedder=Tiny())
    assert kb.stats()["embedder"] == "tiny-16"
    assert kb.search("growth plan", k=1, mode="vector")  # vectors are comparable again


async def test_retrieval_eval_suite_passes():
    results = await run_suite(ROOT / "evals" / "cases.json", include_agent=False)
    failed = [(r.id, r.checks) for r in results if not r.passed]
    assert not failed, failed
