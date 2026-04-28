"""
tests/test_retriever.py
Unit tests for the RRF merge logic (no models or indexes needed).
Run with: pytest tests/test_retriever.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from retriever import _rrf_score


def test_rrf_score_decreasing():
    """Higher-ranked results should have higher RRF scores."""
    assert _rrf_score(0) > _rrf_score(1) > _rrf_score(5) > _rrf_score(20)


def test_rrf_score_positive():
    """RRF scores should always be positive."""
    for i in range(20):
        assert _rrf_score(i) > 0


def test_rrf_merge_deduplication():
    """
    When the same chunk appears in both vector and BM25 results,
    it should appear only once in the merged output with combined score.
    """
    # Manually test the merge logic without loading models
    from retriever import _rrf_score

    vector_hits = [
        {"id": "a", "source": "doc.pdf", "chunk_index": 0, "text": "alpha"},
        {"id": "b", "source": "doc.pdf", "chunk_index": 1, "text": "beta"},
    ]
    bm25_hits = [
        {"id": "c",   "source": "doc.pdf", "chunk_index": 0, "text": "alpha"},  # same as vector[0]
        {"id": "d",   "source": "doc.pdf", "chunk_index": 2, "text": "gamma"},
    ]

    # Replicate the RRF merge logic
    rrf_scores = {}
    chunk_map  = {}

    def key(h):
        return f"{h['source']}::{h['chunk_index']}"

    for rank, hit in enumerate(vector_hits):
        k = key(hit)
        rrf_scores[k] = rrf_scores.get(k, 0.0) + _rrf_score(rank)
        chunk_map[k]  = hit

    for rank, hit in enumerate(bm25_hits):
        k = key(hit)
        rrf_scores[k] = rrf_scores.get(k, 0.0) + _rrf_score(rank)
        if k not in chunk_map:
            chunk_map[k] = hit

    # chunk 0 appears in both → should have higher score than chunk 1 or 2
    k0 = "doc.pdf::0"
    k1 = "doc.pdf::1"
    k2 = "doc.pdf::2"

    assert k0 in rrf_scores
    assert k1 in rrf_scores
    assert k2 in rrf_scores

    # chunk_index 0 appeared in both lists → should have highest combined score
    assert rrf_scores[k0] > rrf_scores[k1]
    assert rrf_scores[k0] > rrf_scores[k2]


def test_rrf_merge_unique_keys():
    """Unique chunks from each source should all appear in merged output."""
    vector_only = [{"source": "a.pdf", "chunk_index": i, "text": f"v{i}", "id": f"v{i}"} for i in range(3)]
    bm25_only   = [{"source": "b.pdf", "chunk_index": i, "text": f"b{i}", "id": f"b{i}"} for i in range(3)]

    rrf_scores = {}
    chunk_map  = {}

    def key(h):
        return f"{h['source']}::{h['chunk_index']}"

    for rank, hit in enumerate(vector_only):
        k = key(hit)
        rrf_scores[k] = rrf_scores.get(k, 0.0) + _rrf_score(rank)
        chunk_map[k] = hit

    for rank, hit in enumerate(bm25_only):
        k = key(hit)
        rrf_scores[k] = rrf_scores.get(k, 0.0) + _rrf_score(rank)
        if k not in chunk_map:
            chunk_map[k] = hit

    assert len(rrf_scores) == 6, "All 6 unique chunks should appear in merged results"
