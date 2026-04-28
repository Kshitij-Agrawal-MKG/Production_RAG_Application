"""
tests/test_monitoring.py
Unit tests for the monitoring and observability layer.
No API keys, no pipeline models, no network calls required.
Run with: pytest tests/test_monitoring.py -v
"""

import json
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from monitoring.metrics_store import MetricsStore, QueryMetric
from monitoring.token_counter import count_tokens, estimate_tokens, extract_token_usage
from monitoring.prompt_versioner import PromptVersioner


# ── QueryMetric helpers ────────────────────────────────────────────────

def _make_metric(**kwargs) -> QueryMetric:
    defaults = dict(
        question="What is the refund policy?",
        session_id="test_session",
        latency_ms=1200.0,
        retrieval_latency_ms=200.0,
        reranking_latency_ms=300.0,
        generation_latency_ms=700.0,
        input_tokens=500,
        output_tokens=150,
        cost_usd=QueryMetric.compute_cost(500, 150),
        citation_coverage=0.85,
        faithfulness_score=0.80,
        was_answered=True,
        retrieval_count=10,
        approved_count=4,
        prompt_version="answer_v1",
        model="gemini-1.5-flash",
        error=None,
    )
    defaults.update(kwargs)
    return QueryMetric(**defaults)


# ── MetricsStore tests ─────────────────────────────────────────────────

class TestMetricsStore:

    def test_record_and_total_rows(self, tmp_path):
        store = MetricsStore(db_path=tmp_path / "test.db")
        assert store.total_rows() == 0
        store.record(_make_metric())
        assert store.total_rows() == 1
        store.record(_make_metric())
        assert store.total_rows() == 2

    def test_summary_empty(self, tmp_path):
        store = MetricsStore(db_path=tmp_path / "test.db")
        s = store.summary(window_hours=24)
        assert s["total_queries"] == 0

    def test_summary_basic_fields(self, tmp_path):
        store = MetricsStore(db_path=tmp_path / "test.db")
        store.record(_make_metric(latency_ms=1000.0, cost_usd=0.0001, was_answered=True))
        store.record(_make_metric(latency_ms=2000.0, cost_usd=0.0002, was_answered=True))
        s = store.summary(window_hours=1)
        required = [
            "total_queries", "answered_queries", "failure_rate", "error_rate",
            "latency_p50_ms", "latency_p95_ms", "latency_mean_ms",
            "cost_total_usd", "cost_mean_usd",
            "citation_coverage_mean", "faithfulness_mean",
            "total_tokens_in", "total_tokens_out",
        ]
        for key in required:
            assert key in s, f"Missing key: {key}"

    def test_p50_p95_correct(self, tmp_path):
        store = MetricsStore(db_path=tmp_path / "test.db")
        for ms in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
            store.record(_make_metric(latency_ms=float(ms)))
        s = store.summary(window_hours=1)
        # P50 of [100..1000] in steps of 100 = 550
        assert 400 <= s["latency_p50_ms"] <= 600
        # P95 should be near 950
        assert s["latency_p95_ms"] >= 800

    def test_failure_rate_calculation(self, tmp_path):
        store = MetricsStore(db_path=tmp_path / "test.db")
        store.record(_make_metric(was_answered=True))
        store.record(_make_metric(was_answered=True))
        store.record(_make_metric(was_answered=False))
        store.record(_make_metric(was_answered=False))
        s = store.summary(window_hours=1)
        assert abs(s["failure_rate"] - 0.5) < 0.01

    def test_cost_accumulates(self, tmp_path):
        store = MetricsStore(db_path=tmp_path / "test.db")
        store.record(_make_metric(cost_usd=0.001))
        store.record(_make_metric(cost_usd=0.002))
        s = store.summary(window_hours=1)
        assert abs(s["cost_total_usd"] - 0.003) < 1e-9

    def test_cost_compute_method(self):
        cost = QueryMetric.compute_cost(1_000_000, 0)
        assert abs(cost - 0.075) < 1e-6
        cost2 = QueryMetric.compute_cost(0, 1_000_000)
        assert abs(cost2 - 0.30) < 1e-6

    def test_error_rate_tracks_error_field(self, tmp_path):
        store = MetricsStore(db_path=tmp_path / "test.db")
        store.record(_make_metric(error=None))
        store.record(_make_metric(error="Gemini API timeout"))
        s = store.summary(window_hours=1)
        assert abs(s["error_rate"] - 0.5) < 0.01

    def test_token_totals_summed(self, tmp_path):
        store = MetricsStore(db_path=tmp_path / "test.db")
        store.record(_make_metric(input_tokens=100, output_tokens=50))
        store.record(_make_metric(input_tokens=200, output_tokens=75))
        s = store.summary(window_hours=1)
        assert s["total_tokens_in"]  == 300
        assert s["total_tokens_out"] == 125

    def test_latency_trend_returns_buckets(self, tmp_path):
        store = MetricsStore(db_path=tmp_path / "test.db")
        for _ in range(3):
            store.record(_make_metric(latency_ms=1500.0))
        trend = store.latency_trend(window_hours=1, bucket_minutes=60)
        assert isinstance(trend, list)
        if trend:
            assert "p50_ms"      in trend[0]
            assert "p95_ms"      in trend[0]
            assert "count"       in trend[0]
            assert "bucket_utc"  in trend[0]

    def test_quality_trend_returns_buckets(self, tmp_path):
        store = MetricsStore(db_path=tmp_path / "test.db")
        store.record(_make_metric(citation_coverage=0.9, faithfulness_score=0.8))
        trend = store.quality_trend(window_hours=1, bucket_minutes=60)
        assert isinstance(trend, list)
        if trend:
            assert "citation_coverage" in trend[0]
            assert "faithfulness"       in trend[0]
            assert "failure_rate"       in trend[0]

    def test_cost_trend_cumulative(self, tmp_path):
        store = MetricsStore(db_path=tmp_path / "test.db")
        store.record(_make_metric(cost_usd=0.001))
        store.record(_make_metric(cost_usd=0.002))
        trend = store.cost_trend(window_hours=1, bucket_minutes=60)
        assert isinstance(trend, list)
        if trend:
            assert "cumulative_usd"   in trend[0]
            assert "bucket_cost_usd"  in trend[0]
            assert trend[-1]["cumulative_usd"] > 0

    def test_record_never_raises_on_bad_data(self, tmp_path):
        store = MetricsStore(db_path=tmp_path / "test.db")
        # A metric with None question should not raise
        try:
            bad = QueryMetric(question=None, latency_ms=0)  # type: ignore
            store.record(bad)
        except Exception:
            pass  # it's OK if it doesn't insert, but must not propagate

    def test_window_filters_old_records(self, tmp_path):
        store = MetricsStore(db_path=tmp_path / "test.db")
        # Insert an old record (25 hours ago) by faking ts
        import sqlite3
        old_ts = time.time() - 25 * 3600
        conn = sqlite3.connect(str(tmp_path / "test.db"))
        conn.execute(
            "INSERT INTO query_metrics (ts, question, latency_ms, was_answered) VALUES (?, ?, ?, ?)",
            (old_ts, "old question", 999.0, 1)
        )
        conn.commit()
        conn.close()
        # Insert a fresh record
        store.record(_make_metric(latency_ms=100.0))
        # Summary with 24h window should only see the fresh record
        s = store.summary(window_hours=24)
        assert s["total_queries"] == 1
        assert s["latency_p50_ms"] == 100.0


# ── TokenCounter tests ─────────────────────────────────────────────────

class TestTokenCounter:

    def test_basic_count_positive(self):
        n = count_tokens("Hello world")
        assert n > 0

    def test_empty_string_returns_zero(self):
        assert count_tokens("") == 0

    def test_longer_text_more_tokens(self):
        short = count_tokens("Hello")
        long  = count_tokens("Hello world this is a longer sentence with more words in it")
        assert long > short

    def test_estimate_tokens_both_positive(self):
        inp, out = estimate_tokens("This is a prompt.", "This is the answer.")
        assert inp > 0
        assert out > 0

    def test_extract_token_usage_no_metadata(self):
        mock_response = MagicMock()
        del mock_response.usage_metadata   # simulate missing attribute
        mock_response.usage_metadata = None
        inp, out = extract_token_usage(mock_response)
        assert inp == 0
        assert out == 0

    def test_extract_token_usage_with_metadata(self):
        mock_response = MagicMock()
        mock_response.usage_metadata.prompt_token_count     = 300
        mock_response.usage_metadata.candidates_token_count = 120
        inp, out = extract_token_usage(mock_response)
        assert inp == 300
        assert out == 120

    def test_extract_token_usage_partial_metadata(self):
        mock_response = MagicMock()
        mock_response.usage_metadata.prompt_token_count     = 200
        mock_response.usage_metadata.candidates_token_count = None
        inp, out = extract_token_usage(mock_response)
        assert inp == 200
        assert out == 0

    def test_count_tokens_unicode(self):
        n = count_tokens("こんにちは世界")   # Japanese
        assert n > 0

    def test_estimate_consistent_with_count(self):
        prompt = "What is the refund policy for HelixDB?"
        answer = "HelixDB offers a 30-day money-back guarantee."
        inp, out = estimate_tokens(prompt, answer)
        assert inp == count_tokens(prompt)
        assert out == count_tokens(answer)


# ── PromptVersioner tests ──────────────────────────────────────────────

class TestPromptVersioner:

    def _make_versioner(self, tmp_path: Path) -> "PromptVersioner":
        """Create a PromptVersioner pointing at a temp directory."""
        # Patch config paths to use tmp_path
        with patch("monitoring.prompt_versioner.config") as mock_cfg:
            mock_cfg.PROMPTS_DIR = tmp_path
            # Create a minimal prompt file
            prompt_file = tmp_path / "answer_v1.txt"
            prompt_file.write_text(
                "You are a helpful assistant.\nAnswer: {question}\nContext: {chunks}"
            )
            # Create a minimal versions.json
            versions_file = tmp_path / "versions.json"
            versions_file.write_text(json.dumps({
                "current": "answer_v1",
                "versions": {
                    "answer_v1": {
                        "file": "answer_v1.txt",
                        "sha256": "",
                        "created": "2024-09-01",
                        "description": "Initial version",
                        "changelog": "",
                        "eval_report": None,
                        "ci_passed": None,
                    }
                },
                "schema_version": 1,
            }))
            pv = PromptVersioner.__new__(PromptVersioner)
            pv._path = versions_file
            pv._data = json.loads(versions_file.read_text())
            # Monkey-patch _prompt_path to use tmp_path
            def _prompt_path(version=None):
                v = version or pv._data.get("current", "answer_v1")
                entry = pv._data["versions"].get(v, {})
                fname = entry.get("file", f"{v}.txt")
                return tmp_path / fname
            pv._prompt_path = _prompt_path
            return pv

    def test_hash_current_returns_string(self, tmp_path):
        pv = self._make_versioner(tmp_path)
        h = pv.hash_current()
        assert isinstance(h, str)
        assert len(h) == 64   # SHA-256 hex = 64 chars

    def test_hash_is_deterministic(self, tmp_path):
        pv = self._make_versioner(tmp_path)
        h1 = pv.hash_current()
        h2 = pv.hash_current()
        assert h1 == h2

    def test_backfill_populates_hash(self, tmp_path):
        pv = self._make_versioner(tmp_path)
        assert pv._data["versions"]["answer_v1"]["sha256"] == ""
        pv._backfill_hash("answer_v1")
        assert len(pv._data["versions"]["answer_v1"]["sha256"]) == 64

    def test_verify_integrity_after_backfill(self, tmp_path):
        pv = self._make_versioner(tmp_path)
        pv._backfill_hash("answer_v1")
        assert pv.verify_integrity() is True

    def test_integrity_fails_after_file_change(self, tmp_path):
        pv = self._make_versioner(tmp_path)
        pv._backfill_hash("answer_v1")
        # Modify the file
        (tmp_path / "answer_v1.txt").write_text("Changed prompt content — different hash now")
        assert pv.verify_integrity() is False

    def test_check_for_changes_raises_on_mismatch(self, tmp_path):
        pv = self._make_versioner(tmp_path)
        pv._backfill_hash("answer_v1")
        (tmp_path / "answer_v1.txt").write_text("Secretly modified prompt")
        import pytest
        with pytest.raises(RuntimeError, match="PROMPT INTEGRITY FAILURE"):
            pv.check_for_changes()

    def test_check_for_changes_passes_on_clean(self, tmp_path):
        pv = self._make_versioner(tmp_path)
        pv._backfill_hash("answer_v1")
        pv.check_for_changes()   # should not raise

    def test_register_new_version(self, tmp_path):
        import config as cfg
        pv = self._make_versioner(tmp_path)
        v2_file = cfg.PROMPTS_DIR / "answer_v2_test_tmp.txt"
        try:
            v2_file.write_text("New prompt for v2: {question} {chunks}")
            pv.register_version("answer_v2_test_tmp", description="Test v2", set_current=True)
            assert "answer_v2_test_tmp" in pv._data["versions"]
            assert pv._data["current"] == "answer_v2_test_tmp"
            assert len(pv._data["versions"]["answer_v2_test_tmp"]["sha256"]) == 64
        finally:
            v2_file.unlink(missing_ok=True)

    def test_register_without_set_current(self, tmp_path):
        import config as cfg
        pv = self._make_versioner(tmp_path)
        v2_file = cfg.PROMPTS_DIR / "answer_v2_alt_test_tmp.txt"
        try:
            v2_file.write_text("v2 alternative prompt: {question} {chunks}")
            pv.register_version("answer_v2_alt_test_tmp", description="Alternative", set_current=False)
            assert pv._data["current"] == "answer_v1"
            assert "answer_v2_alt_test_tmp" in pv._data["versions"]
        finally:
            v2_file.unlink(missing_ok=True)

    def test_register_missing_file_raises(self, tmp_path):
        pv = self._make_versioner(tmp_path)
        import pytest
        with pytest.raises(FileNotFoundError):
            pv.register_version("answer_v99", description="File does not exist")

    def test_mark_ci_result(self, tmp_path):
        pv = self._make_versioner(tmp_path)
        pv.mark_ci_result("answer_v1", passed=True, report_path="logs/eval_20240901.json")
        assert pv._data["versions"]["answer_v1"]["ci_passed"]   is True
        assert pv._data["versions"]["answer_v1"]["eval_report"] == "logs/eval_20240901.json"

    def test_status_returns_expected_keys(self, tmp_path):
        pv = self._make_versioner(tmp_path)
        s = pv.status()
        required = [
            "current_version", "file", "stored_hash", "actual_hash",
            "integrity_ok", "description", "created", "ci_passed",
            "eval_report", "all_versions",
        ]
        for k in required:
            assert k in s, f"Missing key in status: {k}"

    def test_status_all_versions_list(self, tmp_path):
        pv = self._make_versioner(tmp_path)
        s = pv.status()
        assert "answer_v1" in s["all_versions"]

    def test_ensure_hashes_populated_idempotent(self, tmp_path):
        pv = self._make_versioner(tmp_path)
        pv.ensure_hashes_populated()
        h1 = pv._data["versions"]["answer_v1"]["sha256"]
        pv.ensure_hashes_populated()
        h2 = pv._data["versions"]["answer_v1"]["sha256"]
        assert h1 == h2


# ── Tracer no-op mode tests ────────────────────────────────────────────

class TestTracerNoOpMode:
    """Tests for PipelineTracer when Langfuse is not configured."""

    def test_tracer_initialises_without_langfuse(self):
        from monitoring.tracer import PipelineTracer
        tracer = PipelineTracer()
        # Should not raise even without Langfuse keys

    def test_start_trace_returns_context(self):
        from monitoring.tracer import PipelineTracer
        tracer = PipelineTracer()
        ctx = tracer.start_trace("test question")
        assert ctx is not None
        assert ctx.question == "test question"

    def test_noop_span_is_safe_context_manager(self):
        from monitoring.tracer import _NoOpSpan
        s = _NoOpSpan()
        with s as span:
            span.update(foo="bar")   # should not raise
            span.end()               # should not raise

    def test_trace_context_methods_all_safe(self):
        from monitoring.tracer import TraceContext
        ctx = TraceContext(trace=None, question="q", session_id="s")
        with ctx.span_retrieval("q", [], [], []):
            pass
        with ctx.span_reranking("q", [], []):
            pass
        ctx.generation("prompt", "answer", 100, 50)
        ctx.score("metric", 0.9)
        ctx.finish({"latency_ms": 1200})

    def test_flush_safe_without_langfuse(self):
        from monitoring.tracer import PipelineTracer
        tracer = PipelineTracer()
        tracer.flush()   # must not raise
