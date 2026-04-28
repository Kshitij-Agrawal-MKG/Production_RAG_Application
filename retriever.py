"""
retriever.py
Hybrid retrieval: BM25 (keyword) + vector similarity, fused with RRF.

Flow:
  query → BM25 top-k  ┐
                       ├── RRF merge → combined ranked list
  query → Vector top-k ┘

Usage:
  from retriever import HybridRetriever
  retriever = HybridRetriever()
  results = retriever.retrieve("What is the refund policy?", top_k=10)
"""

import pickle
from pathlib import Path
from typing import List, Dict

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from rich.console import Console

import config

console = Console()


def _rrf_score(rank: int, k: int = config.RRF_K) -> float:
    """Reciprocal Rank Fusion score for a result at position `rank` (0-indexed)."""
    return 1.0 / (k + rank + 1)


class HybridRetriever:
    """
    Combines BM25 keyword search and vector similarity search.
    Results are fused with Reciprocal Rank Fusion (RRF).
    """

    def __init__(self):
        # ── Vector store ───────────────────────────────────────────────
        if not config.CHROMA_DIR.exists():
            raise FileNotFoundError(
                f"ChromaDB not found at {config.CHROMA_DIR}. "
                "Run ingestion first: python cli.py ingest <path>"
            )
        self.chroma_client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        self.collection = self.chroma_client.get_or_create_collection(
            name=config.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

        # ── Embedding model ────────────────────────────────────────────
        console.print("[dim]Loading embedding model…[/dim]")
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL)

        # ── BM25 index ────────────────────────────────────────────────
        if not config.BM25_INDEX_PATH.exists():
            raise FileNotFoundError(
                f"BM25 index not found at {config.BM25_INDEX_PATH}. "
                "Run ingestion first."
            )
        console.print("[dim]Loading BM25 index…[/dim]")
        with open(config.BM25_INDEX_PATH, "rb") as f:
            data = pickle.load(f)
        self._bm25_corpus   = data["corpus"]    # List[List[str]]
        self._bm25_metadata = data["metadata"]  # List[Dict]
        self.bm25 = BM25Okapi(self._bm25_corpus)

    # ── Individual retrieval methods ───────────────────────────────────

    def _vector_search(self, query: str, top_k: int) -> List[Dict]:
        """
        Returns list of dicts:
          {id, text, source, chunk_index, score (cosine similarity)}
        """
        if self.collection.count() == 0:
            return []

        query_embedding = self.embedder.encode(
            query, normalize_embeddings=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for i, doc_id in enumerate(results["ids"][0]):
            # ChromaDB cosine distance → similarity: sim = 1 - distance
            distance = results["distances"][0][i]
            similarity = 1.0 - distance
            hits.append({
                "id":          doc_id,
                "text":        results["documents"][0][i],
                "source":      results["metadatas"][0][i].get("source", "unknown"),
                "chunk_index": results["metadatas"][0][i].get("chunk_index", 0),
                "score":       similarity,
            })
        return hits

    def _bm25_search(self, query: str, top_k: int) -> List[Dict]:
        """
        Returns list of dicts:
          {id, text, source, chunk_index, score (BM25 raw)}
        """
        if not self._bm25_corpus:
            return []

        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        hits = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] == 0:
                break  # BM25 score of 0 means no keyword overlap
            meta = self._bm25_metadata[idx]
            doc_id = f"bm25_{meta['source']}_{meta['chunk_index']}"
            hits.append({
                "id":          doc_id,
                "text":        meta["text"],
                "source":      meta["source"],
                "chunk_index": meta["chunk_index"],
                "score":       float(scores[idx]),
            })
        return hits

    # ── RRF fusion ─────────────────────────────────────────────────────

    def _rrf_merge(
        self,
        vector_hits: List[Dict],
        bm25_hits: List[Dict],
    ) -> List[Dict]:
        """
        Merge two ranked lists with Reciprocal Rank Fusion.
        De-duplicates by (source, chunk_index).
        Returns list sorted by descending RRF score.
        """
        rrf_scores: Dict[str, float] = {}
        chunk_map:  Dict[str, Dict]  = {}

        def _key(hit: Dict) -> str:
            return f"{hit['source']}::{hit['chunk_index']}"

        for rank, hit in enumerate(vector_hits):
            k = _key(hit)
            rrf_scores[k] = rrf_scores.get(k, 0.0) + _rrf_score(rank)
            chunk_map[k]  = hit

        for rank, hit in enumerate(bm25_hits):
            k = _key(hit)
            rrf_scores[k] = rrf_scores.get(k, 0.0) + _rrf_score(rank)
            if k not in chunk_map:
                chunk_map[k] = hit

        merged = []
        for k, rrf in sorted(rrf_scores.items(), key=lambda x: -x[1]):
            entry = dict(chunk_map[k])
            entry["rrf_score"] = rrf
            merged.append(entry)

        return merged

    # ── Public API ─────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        vector_k: int = config.VECTOR_TOP_K,
        bm25_k:   int = config.BM25_TOP_K,
    ) -> List[Dict]:
        """
        Main retrieval entry point.
        Returns merged, RRF-ranked list of chunk dicts.
        Each dict has: id, text, source, chunk_index, rrf_score.
        """
        vector_hits = self._vector_search(query, top_k=vector_k)
        bm25_hits   = self._bm25_search(query,   top_k=bm25_k)
        merged      = self._rrf_merge(vector_hits, bm25_hits)
        return merged
