"""
evaluation/metrics.py
Offline evaluation metrics for the RAG pipeline.

Metrics implemented (no external API calls — all local):
  - retrieval_precision     : fraction of retrieved chunks that are relevant
  - retrieval_recall        : fraction of expected keywords found in top chunks
  - citation_coverage       : fraction of answer sentences that have a citation
  - faithfulness_score      : fraction of answer claims supported by chunks
  - hallucination_flag      : True if answer contains content not in any chunk
  - answered_correctly      : True if expected keyword found in answer
  - unanswerable_respected  : True if system said "could not find" for unanswerable Q
  - latency_ms              : time taken for the full pipeline query

All metrics return values in [0.0, 1.0] unless noted.
"""

import re
import time
from typing import List, Dict, Optional, Tuple


# ── Helpers ────────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    return text.lower().strip()


def _sentences(text: str) -> List[str]:
    """Split text into sentences (simple regex — avoids heavy NLP deps)."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in parts if s.strip()]


def _keyword_in_text(keyword: str, text: str) -> bool:
    """Case-insensitive substring match (stem-tolerant via 4-char prefix)."""
    kw = _normalise(keyword)
    txt = _normalise(text)
    # exact match
    if kw in txt:
        return True
    # prefix match (covers "refund" matching "refunds", "refunded" etc.)
    if len(kw) >= 4 and any(kw[:4] in word for word in txt.split()):
        return True
    return False


# ── Individual metric functions ────────────────────────────────────────

def retrieval_recall(
    expected_keywords: List[str],
    retrieved_chunks: List[Dict],
) -> float:
    """
    What fraction of expected keywords are found in any retrieved chunk?
    Measures whether the retrieval system fetched relevant material at all.
    """
    if not expected_keywords:
        return 1.0  # vacuously true — nothing expected

    all_chunk_text = " ".join(c["text"] for c in retrieved_chunks)
    hits = sum(1 for kw in expected_keywords if _keyword_in_text(kw, all_chunk_text))
    return hits / len(expected_keywords)


def retrieval_precision(
    expected_keywords: List[str],
    retrieved_chunks: List[Dict],
    top_k: int = 5,
) -> float:
    """
    Of the top-k retrieved chunks, what fraction contain at least one
    expected keyword?
    """
    if not expected_keywords or not retrieved_chunks:
        return 0.0

    top = retrieved_chunks[:top_k]
    relevant = sum(
        1 for c in top
        if any(_keyword_in_text(kw, c["text"]) for kw in expected_keywords)
    )
    return relevant / len(top)


def citation_coverage(answer: str) -> float:
    """
    Fraction of answer sentences that contain at least one [chunk_N] citation.
    A well-grounded answer should cite every factual sentence.
    """
    sents = _sentences(answer)
    if not sents:
        return 0.0

    cited = sum(1 for s in sents if re.search(r"\[chunk_\d+\]", s, re.IGNORECASE))
    return cited / len(sents)


def faithfulness_score(
    answer: str,
    approved_chunks: List[Dict],
) -> float:
    """
    Fraction of answer sentences whose key nouns/terms appear in the
    approved (grounding) chunks.

    This is a lightweight local proxy for RAGas-style faithfulness.
    It checks if the vocabulary of each answer sentence overlaps with the
    grounding context — it won't catch every hallucination but flags obvious ones.
    """
    sents = _sentences(answer)
    if not sents or not approved_chunks:
        return 0.0

    # Build a bag-of-words from all approved chunks
    chunk_words = set()
    for c in approved_chunks:
        for w in _normalise(c["text"]).split():
            if len(w) >= 4:   # skip stop-word-length tokens
                chunk_words.add(w)

    supported = 0
    for sent in sents:
        # Skip meta-sentences like "I could not find..."
        if re.search(r"could not find|no reliable|not enough", sent, re.IGNORECASE):
            supported += 1
            continue
        # Remove citation markers before checking
        clean = re.sub(r"\[chunk_\d+\]", "", sent)
        sent_words = [w for w in _normalise(clean).split() if len(w) >= 4]
        if not sent_words:
            supported += 1
            continue
        overlap = sum(1 for w in sent_words if w in chunk_words)
        coverage = overlap / len(sent_words)
        if coverage >= 0.35:   # at least 35% word overlap = "supported"
            supported += 1

    return supported / len(sents)


def hallucination_flag(
    answer: str,
    approved_chunks: List[Dict],
) -> bool:
    """
    Returns True if a sentence in the answer has very low grounding
    overlap with the approved chunks (potential hallucination signal).
    Conservative — only flags extreme cases (< 20% word overlap).
    """
    sents = _sentences(answer)
    if not approved_chunks:
        return False

    chunk_words = set()
    for c in approved_chunks:
        for w in _normalise(c["text"]).split():
            if len(w) >= 4:
                chunk_words.add(w)

    for sent in sents:
        if re.search(r"could not find|no reliable|not enough|\[chunk_", sent, re.IGNORECASE):
            continue
        clean = re.sub(r"\[chunk_\d+\]", "", sent)
        words = [w for w in _normalise(clean).split() if len(w) >= 4]
        if len(words) < 5:
            continue  # too short to judge
        overlap = sum(1 for w in words if w in chunk_words)
        if overlap / len(words) < 0.20:
            return True   # low overlap → possible hallucination
    return False


def answered_correctly(
    answer: str,
    expected_keywords: List[str],
    expected_answer: Optional[str],
) -> bool:
    """
    Did the system produce a substantive answer that contains
    at least one expected keyword?
    Returns True if:
      - expected_answer is None (unanswerable) AND system said "could not find"
      - expected_answer is not None AND at least one expected_keyword in answer
    """
    is_unanswerable = expected_answer is None

    no_answer_phrases = ["could not find", "no reliable", "not enough information",
                         "cannot answer", "do not have"]
    said_no_answer = any(p in answer.lower() for p in no_answer_phrases)

    if is_unanswerable:
        return said_no_answer

    # For answerable questions: at least one keyword must be in the answer
    if not expected_keywords:
        return not said_no_answer
    return any(_keyword_in_text(kw, answer) for kw in expected_keywords)


def unanswerable_respected(
    answer: str,
    expected_answer: Optional[str],
) -> Optional[bool]:
    """
    For unanswerable questions only — did the system correctly abstain?
    Returns None for answerable questions (not applicable).
    """
    if expected_answer is not None:
        return None  # N/A

    no_answer_phrases = ["could not find", "no reliable", "not enough information",
                         "cannot answer", "do not have"]
    return any(p in answer.lower() for p in no_answer_phrases)


def compute_all_metrics(
    question: str,
    answer: str,
    expected_keywords: List[str],
    expected_answer: Optional[str],
    retrieved_chunks: List[Dict],
    approved_chunks: List[Dict],
    latency_ms: float,
) -> Dict:
    """
    Compute all metrics for a single Q&A pair.
    Returns a flat dict with all metric values.
    """
    ret_recall    = retrieval_recall(expected_keywords, retrieved_chunks)
    ret_precision = retrieval_precision(expected_keywords, retrieved_chunks)
    cit_cov       = citation_coverage(answer)
    faith         = faithfulness_score(answer, approved_chunks)
    halluc        = hallucination_flag(answer, approved_chunks)
    correct       = answered_correctly(answer, expected_keywords, expected_answer)
    unans         = unanswerable_respected(answer, expected_answer)

    return {
        "retrieval_recall":       round(ret_recall,    4),
        "retrieval_precision":    round(ret_precision, 4),
        "citation_coverage":      round(cit_cov,       4),
        "faithfulness_score":     round(faith,         4),
        "hallucination_flag":     halluc,
        "answered_correctly":     correct,
        "unanswerable_respected": unans,
        "latency_ms":             round(latency_ms,    1),
    }
