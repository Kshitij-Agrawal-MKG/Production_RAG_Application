"""
config.py
Central configuration for the Ask My Docs RAG system.
All tunable knobs live here — change once, applies everywhere.
"""

from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data"
DOCS_DIR        = DATA_DIR / "docs"
INDEX_DIR       = DATA_DIR / "indexes"
PROMPTS_DIR     = BASE_DIR / "prompts"
CHROMA_DIR      = INDEX_DIR / "chroma"
BM25_INDEX_PATH = INDEX_DIR / "bm25_index.pkl"

# ── API Keys ───────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ── Chunking ───────────────────────────────────────────────────────────
CHUNK_SIZE        = 600    # target tokens per chunk
CHUNK_OVERLAP     = 100    # token overlap between adjacent chunks
MIN_CHUNK_TOKENS  = 50     # discard chunks shorter than this

# ── Embedding model ────────────────────────────────────────────────────
EMBEDDING_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"

# ── Retrieval ──────────────────────────────────────────────────────────
VECTOR_TOP_K      = 10     # candidates from vector search
BM25_TOP_K        = 10     # candidates from BM25 search
RERANK_TOP_K      = 5      # final chunks passed to LLM after reranking

# RRF (Reciprocal Rank Fusion) constant — higher = less weight to top ranks
RRF_K             = 60

# ── Cross-encoder reranker ─────────────────────────────────────────────
RERANKER_MODEL    = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Citation enforcement ───────────────────────────────────────────────
MIN_RELEVANCE_SCORE = 0.0   # cross-encoder score threshold; 0.0 = accept all
                             # raise to e.g. -3.0 to be more strict

# ── Gemini ─────────────────────────────────────────────────────────────
GEMINI_MODEL      = "gemini-3-flash-preview"   # free-tier model
GEMINI_MAX_TOKENS = 1024
GEMINI_TEMPERATURE = 0.2   # low temp for factual grounded answers

# ── ChromaDB ───────────────────────────────────────────────────────────
CHROMA_COLLECTION = "ask_my_docs"

# ── Supported file extensions ──────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".markdown", ".html", ".htm", ".txt"}
