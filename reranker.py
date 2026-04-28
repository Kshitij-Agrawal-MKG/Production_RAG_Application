"""
reranker.py
Cross-encoder reranking + citation enforcement gate.

Flow:
  (query, merged_chunks) → cross-encoder scores → sorted top-k
                         → citation guard (threshold check)
                         → approved chunks (or empty list → no answer)

Usage:
  from reranker import Reranker
  reranker = Reranker()
  top_chunks = reranker.rerank(query, merged_chunks, top_k=5)
"""

from typing import List, Dict

from sentence_transformers import CrossEncoder
from rich.console import Console

import config

console = Console()


class Reranker:
    """
    Cross-encoder reranker using ms-marco-MiniLM-L-6-v2.
    Scores each (query, chunk) pair — more accurate than bi-encoder cosine similarity.
    Also acts as the citation enforcement gate.
    """

    def __init__(self):
        console.print("[dim]Loading cross-encoder reranker…[/dim]")
        self.model = CrossEncoder(config.RERANKER_MODEL)

    def rerank(
        self,
        query: str,
        chunks: List[Dict],
        top_k: int = config.RERANK_TOP_K,
    ) -> List[Dict]:
        """
        Score each chunk against the query and return top_k.
        Adds 'rerank_score' field to each chunk dict.
        Returns an empty list if no chunk passes MIN_RELEVANCE_SCORE.
        """
        if not chunks:
            return []

        pairs = [(query, chunk["text"]) for chunk in chunks]
        scores = self.model.predict(pairs)

        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        # Sort descending by reranker score
        ranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)

        # Citation enforcement gate
        top = ranked[:top_k]
        approved = [c for c in top if c["rerank_score"] >= config.MIN_RELEVANCE_SCORE]

        if not approved:
            console.print(
                f"[yellow]Citation guard:[/yellow] No chunks passed the "
                f"relevance threshold ({config.MIN_RELEVANCE_SCORE}). "
                "Answer will be suppressed."
            )
        return approved
