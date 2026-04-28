"""
tests/test_generator.py
Unit tests for citation parsing in the generator (no API calls).
Run with: pytest tests/test_generator.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from generator import _extract_cited_sources, _format_chunks


SAMPLE_CHUNKS = [
    {"source": "data/docs/policy.pdf",  "chunk_index": 2, "rerank_score": 0.85, "text": "Refunds are processed in 5 days."},
    {"source": "data/docs/policy.pdf",  "chunk_index": 3, "rerank_score": 0.72, "text": "Contact support for exceptions."},
    {"source": "data/docs/terms.docx",  "chunk_index": 0, "rerank_score": 0.61, "text": "All sales final after 30 days."},
]


def test_extract_single_citation():
    answer = "Refunds take 5 business days [chunk_1]."
    sources = _extract_cited_sources(answer, SAMPLE_CHUNKS)
    assert len(sources) == 1
    assert sources[0]["source"] == "data/docs/policy.pdf"
    assert sources[0]["chunk_index"] == 2


def test_extract_multiple_citations():
    answer = "See policy [chunk_1] and terms [chunk_3]."
    sources = _extract_cited_sources(answer, SAMPLE_CHUNKS)
    assert len(sources) == 2
    source_files = {Path(s["source"]).name for s in sources}
    assert "policy.pdf" in source_files
    assert "terms.docx"  in source_files


def test_extract_duplicate_citations():
    """Same chunk cited twice should appear only once in sources."""
    answer = "Refunds [chunk_1] are 5 days [chunk_1]."
    sources = _extract_cited_sources(answer, SAMPLE_CHUNKS)
    assert len(sources) == 1


def test_extract_no_citations():
    """Answer with no [chunk_N] markers returns empty list."""
    answer = "I could not find a reliable answer."
    sources = _extract_cited_sources(answer, SAMPLE_CHUNKS)
    assert sources == []


def test_extract_out_of_range_citation():
    """Citation index beyond chunk list is silently ignored."""
    answer = "According to [chunk_99]."
    sources = _extract_cited_sources(answer, SAMPLE_CHUNKS)
    assert sources == []


def test_format_chunks_contains_source():
    """Formatted chunk block should include source filename and chunk index."""
    formatted = _format_chunks(SAMPLE_CHUNKS)
    assert "policy.pdf" in formatted
    assert "terms.docx"  in formatted
    assert "[chunk_1]"   in formatted
    assert "[chunk_3]"   in formatted


def test_format_chunks_numbering():
    """Chunks are numbered starting from 1."""
    formatted = _format_chunks(SAMPLE_CHUNKS)
    assert "[chunk_1]" in formatted
    assert "[chunk_2]" in formatted
    assert "[chunk_3]" in formatted
    assert "[chunk_0]" not in formatted


def test_format_chunks_empty():
    """Empty chunk list returns empty string."""
    formatted = _format_chunks([])
    assert formatted == ""
