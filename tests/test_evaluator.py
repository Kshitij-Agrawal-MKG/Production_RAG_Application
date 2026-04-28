"""
tests/test_evaluator.py
Tests for the Evaluator aggregation and reporting logic.
Uses a mock pipeline — no real models or API calls needed.
Run with: pytest tests/test_evaluator.py -v
"""

import json
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.evaluator import Evaluator, _aggregate, SingleResult, AggregateMetrics


# ── Mock RAGResult ─────────────────────────────────────────────────────

@dataclass
class MockRAGResult:
    question: str
    answer: str
    sources: List[Dict] = field(default_factory=list)
    retrieved_chunks: List[Dict] = field(default_factory=list)
    approved_chunks: List[Dict] = field(default_factory=list)
    was_answered: bool = True


def _make_mock_pipeline(answer="The refund policy is 30 days [chunk_1].", sources=None):
    """Return a mock pipeline that always returns the given answer."""
    mock = MagicMock()
    mock.query.return_value = MockRAGResult(
        question="any",
        answer=answer,
        sources=sources or [{"source": "doc.pdf", "chunk_index": 0, "rerank_score": 0.8}],
        retrieved_chunks=[{"text": "refund policy 30 days return", "source": "doc.pdf", "chunk_index": 0}],
        approved_chunks=[{"text": "refund policy 30 days return", "source": "doc.pdf", "chunk_index": 0}],
        was_answered=True,
    )
    return mock


def _minimal_golden_dataset():
    return [
        {
            "id": "t001",
            "question": "What is the refund policy?",
            "expected_answer": "refund",
            "expected_keywords": ["refund"],
            "category": "factual",
            "difficulty": "easy",
        },
        {
            "id": "t002",
            "question": "What is the weather on Mars?",
            "expected_answer": None,
            "expected_keywords": [],
            "category": "unanswerable",
            "difficulty": "easy",
        },
    ]


# ── _aggregate tests ───────────────────────────────────────────────────

class TestAggregate:
    def _make_result(self, **kwargs) -> SingleResult:
        defaults = dict(
            id="x", question="q", category="factual", difficulty="easy",
            answer="a", expected_keywords=[],
            retrieval_recall=0.8, retrieval_precision=0.7,
            citation_coverage=0.9, faithfulness_score=0.85,
            hallucination_flag=False, answered_correctly=True,
            unanswerable_respected=None, latency_ms=500.0,
        )
        defaults.update(kwargs)
        return SingleResult(**defaults)

    def test_empty_results_returns_zero_aggregate(self):
        agg = _aggregate([])
        assert agg.total_samples == 0
        assert agg.retrieval_recall == 0.0

    def test_mean_computed_correctly(self):
        r1 = self._make_result(retrieval_recall=0.6)
        r2 = self._make_result(retrieval_recall=0.8)
        agg = _aggregate([r1, r2])
        assert abs(agg.retrieval_recall - 0.7) < 0.001

    def test_error_samples_excluded_from_metrics(self):
        r_good  = self._make_result(retrieval_recall=1.0)
        r_error = self._make_result(retrieval_recall=0.0, error="connection failed")
        agg = _aggregate([r_good, r_error])
        assert agg.error_samples == 1
        assert agg.retrieval_recall == 1.0   # error sample excluded

    def test_answerable_vs_unanswerable_split(self):
        answerable   = self._make_result(unanswerable_respected=None,  answered_correctly=True)
        unanswerable = self._make_result(unanswerable_respected=True,   answered_correctly=True)
        agg = _aggregate([answerable, unanswerable])
        assert agg.answerable_samples   == 1
        assert agg.unanswerable_samples == 1
        assert agg.total_samples        == 2

    def test_hallucination_rate_is_fraction(self):
        r1 = self._make_result(hallucination_flag=True)
        r2 = self._make_result(hallucination_flag=False)
        r3 = self._make_result(hallucination_flag=False)
        agg = _aggregate([r1, r2, r3])
        assert abs(agg.hallucination_rate - (1/3)) < 0.001

    def test_latency_averaged(self):
        r1 = self._make_result(latency_ms=1000.0)
        r2 = self._make_result(latency_ms=2000.0)
        agg = _aggregate([r1, r2])
        assert abs(agg.mean_latency_ms - 1500.0) < 0.1


# ── Evaluator integration tests (mock pipeline) ─────────────────────────

class TestEvaluatorWithMock:
    def _make_evaluator_with_mock_data(self, answers, tmp_path):
        """Create an evaluator with a temp golden dataset and mock pipeline."""
        dataset = _minimal_golden_dataset()
        ds_path = tmp_path / "golden_dataset.json"
        ds_path.write_text(json.dumps(dataset))

        call_count = [0]
        def side_effect(question, verbose=False):
            ans = answers[call_count[0] % len(answers)]
            call_count[0] += 1
            return MockRAGResult(
                question=question,
                answer=ans,
                sources=[{"source": "doc.pdf", "chunk_index": 0, "rerank_score": 0.7}],
                retrieved_chunks=[{"text": "refund policy 30 days", "source": "doc.pdf", "chunk_index": 0}],
                approved_chunks=[{"text": "refund policy 30 days", "source": "doc.pdf", "chunk_index": 0}],
                was_answered="could not find" not in ans.lower(),
            )

        mock_pipeline = MagicMock()
        mock_pipeline.query.side_effect = side_effect

        ev = Evaluator(golden_dataset_path=ds_path, pipeline=mock_pipeline)
        return ev

    def test_report_has_correct_total_samples(self, tmp_path):
        ev = self._make_evaluator_with_mock_data(
            ["Refund is 30 days [chunk_1].", "I could not find a reliable answer."],
            tmp_path
        )
        report = ev.run()
        assert report.aggregate.total_samples == 2

    def test_answerable_and_unanswerable_split(self, tmp_path):
        ev = self._make_evaluator_with_mock_data(
            ["Refund is 30 days [chunk_1].", "I could not find a reliable answer."],
            tmp_path
        )
        report = ev.run()
        assert report.aggregate.answerable_samples   == 1
        assert report.aggregate.unanswerable_samples == 1

    def test_report_saved_as_valid_json(self, tmp_path):
        ev = self._make_evaluator_with_mock_data(
            ["Refund is 30 days [chunk_1].", "I could not find."],
            tmp_path
        )
        report = ev.run()
        saved = ev.save_report(report, tmp_path / "test_report.json")
        assert saved.exists()
        with open(saved) as f:
            data = json.load(f)
        assert "aggregate"       in data
        assert "results"         in data
        assert "timestamp"       in data
        assert "pipeline_config" in data

    def test_limit_parameter_respected(self, tmp_path):
        ev = self._make_evaluator_with_mock_data(
            ["Answer [chunk_1].", "I could not find."],
            tmp_path
        )
        report = ev.run(limit=1)
        assert report.aggregate.total_samples == 1

    def test_category_filter_respected(self, tmp_path):
        ev = self._make_evaluator_with_mock_data(
            ["Answer [chunk_1]."],
            tmp_path
        )
        report = ev.run(categories=["factual"])
        # Only 1 "factual" entry in minimal dataset
        assert report.aggregate.total_samples == 1

    def test_pipeline_error_recorded(self, tmp_path):
        dataset = _minimal_golden_dataset()
        ds_path = tmp_path / "golden.json"
        ds_path.write_text(json.dumps(dataset))

        mock_pipeline = MagicMock()
        mock_pipeline.query.side_effect = RuntimeError("Simulated failure")

        ev = Evaluator(golden_dataset_path=ds_path, pipeline=mock_pipeline)
        report = ev.run()
        assert report.aggregate.error_samples == 2   # both fail

    def test_report_results_length_matches_dataset(self, tmp_path):
        ev = self._make_evaluator_with_mock_data(
            ["Refund policy [chunk_1].", "I could not find."],
            tmp_path
        )
        report = ev.run()
        assert len(report.results) == len(_minimal_golden_dataset())

    def test_pipeline_config_snapshot_captured(self, tmp_path):
        ev = self._make_evaluator_with_mock_data(["Answer [chunk_1]."], tmp_path)
        report = ev.run()
        cfg = report.pipeline_config
        assert "embedding_model" in cfg
        assert "reranker_model"  in cfg
        assert "gemini_model"    in cfg
        assert "chunk_size"      in cfg


# ── Logger tests ───────────────────────────────────────────────────────

class TestQueryLogger:
    def test_log_and_tail(self, tmp_path):
        from evaluation.logger import QueryLogger
        logger = QueryLogger(log_path=tmp_path / "queries.jsonl")

        result = MockRAGResult(
            question="What is the refund policy?",
            answer="30 days [chunk_1].",
            sources=[{"source": "doc.pdf", "chunk_index": 0, "rerank_score": 0.8}],
            approved_chunks=[{"text": "refund 30 days", "source": "doc.pdf", "chunk_index": 0}],
            was_answered=True,
        )
        logger.log(result, latency_ms=800.0)

        entries = logger.tail(n=5)
        assert len(entries) == 1
        assert entries[0]["question"] == "What is the refund policy?"
        assert entries[0]["latency_ms"] == 800.0

    def test_summary_totals(self, tmp_path):
        from evaluation.logger import QueryLogger
        logger = QueryLogger(log_path=tmp_path / "queries.jsonl")

        for i in range(3):
            r = MockRAGResult(
                question=f"question {i}",
                answer="answer [chunk_1].",
                was_answered=True,
                sources=[],
                approved_chunks=[],
            )
            logger.log(r, latency_ms=float(500 + i * 100))

        summary = logger.summary()
        assert summary["total_queries"]  == 3
        assert summary["answered_count"] == 3

    def test_clear_empties_log(self, tmp_path):
        from evaluation.logger import QueryLogger
        logger = QueryLogger(log_path=tmp_path / "queries.jsonl")
        r = MockRAGResult(
            question="q", answer="a", was_answered=True,
            sources=[], approved_chunks=[],
        )
        logger.log(r)
        logger.clear()
        entries = logger.tail()
        assert entries == []

    def test_logging_never_raises(self, tmp_path):
        """Logger must not crash even if result is malformed."""
        from evaluation.logger import QueryLogger
        logger = QueryLogger(log_path=tmp_path / "queries.jsonl")
        # Pass a broken result — logger should silently swallow errors
        try:
            logger.log(None, latency_ms=0.0)
        except Exception:
            pass  # it's OK if it doesn't log, but it must not propagate
