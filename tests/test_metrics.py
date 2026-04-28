"""
tests/test_metrics.py
Unit tests for all evaluation metric functions.
No API calls, no models, no pipeline required.
Run with: pytest tests/test_metrics.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.metrics import (
    retrieval_recall,
    retrieval_precision,
    citation_coverage,
    faithfulness_score,
    hallucination_flag,
    answered_correctly,
    unanswerable_respected,
    compute_all_metrics,
)

# ── Sample fixtures ────────────────────────────────────────────────────

CHUNKS_WITH_KEYWORDS = [
    {"text": "The refund policy allows returns within 30 days of purchase.", "source": "policy.pdf", "chunk_index": 0},
    {"text": "Contact support at support@example.com for assistance.", "source": "policy.pdf", "chunk_index": 1},
    {"text": "All prices are listed in USD and include taxes.", "source": "pricing.pdf", "chunk_index": 0},
]

CHUNKS_UNRELATED = [
    {"text": "The quick brown fox jumps over the lazy dog.", "source": "misc.txt", "chunk_index": 0},
    {"text": "Lorem ipsum dolor sit amet consectetur.", "source": "misc.txt", "chunk_index": 1},
]

CITED_ANSWER = (
    "The refund policy allows returns within 30 days [chunk_1]. "
    "You can contact support for exceptions [chunk_2]."
)

UNCITED_ANSWER = (
    "The refund policy allows returns within 30 days. "
    "You can contact support for exceptions."
)

NO_ANSWER_RESPONSE = "I could not find a reliable answer in the provided documents."


# ── retrieval_recall ───────────────────────────────────────────────────

class TestRetrievalRecall:
    def test_all_keywords_found(self):
        score = retrieval_recall(["refund", "return", "policy"], CHUNKS_WITH_KEYWORDS)
        assert score == 1.0

    def test_no_keywords_found(self):
        score = retrieval_recall(["blockchain", "quantum", "nft"], CHUNKS_WITH_KEYWORDS)
        assert score == 0.0

    def test_partial_keywords(self):
        score = retrieval_recall(["refund", "blockchain"], CHUNKS_WITH_KEYWORDS)
        assert score == 0.5

    def test_empty_keywords_returns_one(self):
        score = retrieval_recall([], CHUNKS_WITH_KEYWORDS)
        assert score == 1.0

    def test_empty_chunks_returns_zero(self):
        score = retrieval_recall(["refund"], [])
        assert score == 0.0

    def test_prefix_matching(self):
        # "refunds" should match keyword "refund" via prefix
        chunks = [{"text": "All refunds are processed in 5 business days.", "source": "x", "chunk_index": 0}]
        score = retrieval_recall(["refund"], chunks)
        assert score == 1.0

    def test_case_insensitive(self):
        chunks = [{"text": "REFUND POLICY IS HERE", "source": "x", "chunk_index": 0}]
        score = retrieval_recall(["refund"], chunks)
        assert score == 1.0


# ── retrieval_precision ────────────────────────────────────────────────

class TestRetrievalPrecision:
    def test_all_relevant(self):
        score = retrieval_precision(["refund"], CHUNKS_WITH_KEYWORDS, top_k=3)
        # All 3 chunks: chunk 0 has "refund" → 1/3 = 0.33
        assert score == pytest_approx_or_near(1/3, 0.01)

    def test_no_relevant(self):
        score = retrieval_precision(["quantum"], CHUNKS_WITH_KEYWORDS, top_k=3)
        assert score == 0.0

    def test_empty_chunks(self):
        score = retrieval_precision(["refund"], [], top_k=5)
        assert score == 0.0

    def test_empty_keywords(self):
        score = retrieval_precision([], CHUNKS_WITH_KEYWORDS, top_k=3)
        assert score == 0.0

    def test_top_k_respected(self):
        # Relevant chunk at index 0 only; top_k=1 should give 1.0
        score = retrieval_precision(["refund"], CHUNKS_WITH_KEYWORDS, top_k=1)
        assert score == 1.0


def pytest_approx_or_near(val, tol):
    """Simple tolerance comparison helper."""
    class _Near:
        def __eq__(self, other):
            return abs(other - val) <= tol
    return _Near()


# ── citation_coverage ──────────────────────────────────────────────────

class TestCitationCoverage:
    def test_fully_cited(self):
        answer = "All claims are cited [chunk_1]. Every sentence has a reference [chunk_2]."
        score = citation_coverage(answer)
        assert score == 1.0

    def test_not_cited(self):
        score = citation_coverage(UNCITED_ANSWER)
        assert score == 0.0

    def test_partially_cited(self):
        answer = "First claim [chunk_1]. Second claim without citation."
        score = citation_coverage(answer)
        assert score == 0.5

    def test_empty_answer(self):
        score = citation_coverage("")
        assert score == 0.0

    def test_chunk_pattern_recognition(self):
        answer = "Some fact [chunk_3]. Another fact [chunk_10]."
        score = citation_coverage(answer)
        assert score == 1.0


# ── faithfulness_score ─────────────────────────────────────────────────

class TestFaithfulnessScore:
    def test_high_overlap_answer(self):
        # Answer uses same words as the chunks
        answer = "The refund policy allows returns within 30 days of purchase [chunk_1]."
        score = faithfulness_score(answer, CHUNKS_WITH_KEYWORDS)
        assert score >= 0.7

    def test_no_chunks(self):
        score = faithfulness_score("Some answer.", [])
        assert score == 0.0

    def test_empty_answer(self):
        score = faithfulness_score("", CHUNKS_WITH_KEYWORDS)
        assert score == 0.0

    def test_no_answer_sentence_is_supported(self):
        # "could not find" sentences are auto-counted as supported
        score = faithfulness_score(NO_ANSWER_RESPONSE, CHUNKS_WITH_KEYWORDS)
        assert score == 1.0

    def test_score_range(self):
        score = faithfulness_score(CITED_ANSWER, CHUNKS_WITH_KEYWORDS)
        assert 0.0 <= score <= 1.0


# ── hallucination_flag ─────────────────────────────────────────────────

class TestHallucinationFlag:
    def test_grounded_answer_no_flag(self):
        # Answer vocabulary overlaps well with chunk vocabulary
        answer = "The refund policy allows returns within 30 days. Contact support for help."
        flagged = hallucination_flag(answer, CHUNKS_WITH_KEYWORDS)
        assert flagged is False

    def test_no_chunks_no_flag(self):
        # No chunks means we can't judge
        flagged = hallucination_flag("Some answer here.", [])
        assert flagged is False

    def test_short_sentences_not_flagged(self):
        # Sentences < 5 words are skipped
        flagged = hallucination_flag("Yes. No. Maybe.", CHUNKS_WITH_KEYWORDS)
        assert flagged is False

    def test_citation_sentences_skipped(self):
        flagged = hallucination_flag("[chunk_1] supports this claim.", CHUNKS_WITH_KEYWORDS)
        assert flagged is False


# ── answered_correctly ─────────────────────────────────────────────────

class TestAnsweredCorrectly:
    def test_correct_answerable(self):
        answer = "The refund window is 30 days as per our policy."
        result = answered_correctly(answer, ["refund", "policy"], "refund")
        assert result is True

    def test_wrong_answerable(self):
        answer = "I could not find a reliable answer."
        result = answered_correctly(answer, ["refund", "policy"], "refund")
        assert result is False

    def test_unanswerable_correctly_abstained(self):
        result = answered_correctly(NO_ANSWER_RESPONSE, [], None)
        assert result is True

    def test_unanswerable_incorrectly_answered(self):
        result = answered_correctly("The answer is 42.", [], None)
        assert result is False

    def test_empty_keywords_non_no_answer(self):
        result = answered_correctly("Here is some information.", [], "something")
        assert result is True  # no keywords to check, not a no-answer response


# ── unanswerable_respected ─────────────────────────────────────────────

class TestUnanswerableRespected:
    def test_answerable_returns_none(self):
        result = unanswerable_respected("Some answer.", "expected")
        assert result is None

    def test_unanswerable_correctly_abstained(self):
        result = unanswerable_respected(NO_ANSWER_RESPONSE, None)
        assert result is True

    def test_unanswerable_incorrectly_answered(self):
        result = unanswerable_respected("The answer is definitely X.", None)
        assert result is False

    def test_various_no_answer_phrases(self):
        for phrase in ["could not find", "no reliable", "not enough information", "cannot answer"]:
            assert unanswerable_respected(f"I {phrase} in the documents.", None) is True


# ── compute_all_metrics ────────────────────────────────────────────────

class TestComputeAllMetrics:
    def test_returns_all_keys(self):
        m = compute_all_metrics(
            question="What is the refund policy?",
            answer=CITED_ANSWER,
            expected_keywords=["refund"],
            expected_answer="refund",
            retrieved_chunks=CHUNKS_WITH_KEYWORDS,
            approved_chunks=CHUNKS_WITH_KEYWORDS,
            latency_ms=1200.0,
        )
        required_keys = [
            "retrieval_recall", "retrieval_precision", "citation_coverage",
            "faithfulness_score", "hallucination_flag", "answered_correctly",
            "unanswerable_respected", "latency_ms",
        ]
        for k in required_keys:
            assert k in m, f"Missing key: {k}"

    def test_values_in_range(self):
        m = compute_all_metrics(
            question="test", answer=CITED_ANSWER,
            expected_keywords=["refund"], expected_answer="refund",
            retrieved_chunks=CHUNKS_WITH_KEYWORDS,
            approved_chunks=CHUNKS_WITH_KEYWORDS,
            latency_ms=500.0,
        )
        for key in ["retrieval_recall", "retrieval_precision", "citation_coverage", "faithfulness_score"]:
            assert 0.0 <= m[key] <= 1.0, f"{key} out of range: {m[key]}"

    def test_latency_preserved(self):
        m = compute_all_metrics(
            question="q", answer="a [chunk_1].",
            expected_keywords=[], expected_answer="a",
            retrieved_chunks=[], approved_chunks=[],
            latency_ms=742.5,
        )
        assert m["latency_ms"] == 742.5
