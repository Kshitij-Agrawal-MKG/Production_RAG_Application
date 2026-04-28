"""
tests/test_chunker.py
Unit tests for the Chunker class (no API calls, no models needed).
Run with: pytest tests/test_chunker.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingest import Chunker
import config


def test_basic_chunking():
    """Chunks are produced from normal text."""
    chunker = Chunker(chunk_size=100, overlap=20, min_tokens=5)
    text = "This is a sentence. " * 50
    chunks = chunker.chunk(text, source="test.txt")
    assert len(chunks) > 0, "Should produce at least one chunk"


def test_chunk_fields():
    """Each chunk dict has required fields."""
    chunker = Chunker()
    text = "Hello world. " * 100
    chunks = chunker.chunk(text, source="test.txt")
    for c in chunks:
        assert "text"        in c
        assert "source"      in c
        assert "chunk_index" in c
        assert "token_count" in c
        assert c["source"] == "test.txt"


def test_min_token_filter():
    """Chunks shorter than min_tokens are discarded."""
    chunker = Chunker(chunk_size=100, overlap=10, min_tokens=30)
    # Very short text → should produce 0 or 1 chunk depending on length
    short_text = "Hi."
    chunks = chunker.chunk(short_text, source="tiny.txt")
    for c in chunks:
        assert c["token_count"] >= 30, "All chunks should meet min token threshold"


def test_overlap_produces_context():
    """Adjacent chunks should share some tokens when text is long enough."""
    chunker = Chunker(chunk_size=80, overlap=30, min_tokens=10)
    text = " ".join([f"word{i}" for i in range(300)])
    chunks = chunker.chunk(text, source="overlap.txt")
    if len(chunks) > 1:
        # The end of chunk 0 and start of chunk 1 should share words
        end_words   = set(chunks[0]["text"].split()[-10:])
        start_words = set(chunks[1]["text"].split()[:10])
        # At least some overlap is expected
        assert len(end_words & start_words) > 0, "Adjacent chunks should share overlapping tokens"


def test_long_paragraph_force_split():
    """A single very long paragraph (no breaks) should still be chunked."""
    chunker = Chunker(chunk_size=50, overlap=10, min_tokens=5)
    # 200 words with no paragraph breaks
    long_para = " ".join([f"token{i}" for i in range(200)])
    chunks = chunker.chunk(long_para, source="long.txt")
    assert len(chunks) > 1, "Long paragraph should be force-split into multiple chunks"


def test_empty_text():
    """Empty text produces no chunks."""
    chunker = Chunker()
    chunks = chunker.chunk("", source="empty.txt")
    assert chunks == []


def test_whitespace_only():
    """Whitespace-only text produces no chunks."""
    chunker = Chunker()
    chunks = chunker.chunk("   \n\n\t  ", source="whitespace.txt")
    assert chunks == []


def test_chunk_indices_sequential():
    """Chunk indices within a single source are sequential."""
    chunker = Chunker(chunk_size=60, overlap=10, min_tokens=5)
    text = "sentence number one. " * 100
    chunks = chunker.chunk(text, source="seq.txt")
    indices = [c["chunk_index"] for c in chunks]
    assert indices == list(range(len(chunks))), "chunk_index should be 0,1,2,…"
