"""
tests/test_ci_gate.py
Unit tests for the CI gate threshold logic.
No pipeline, no API calls required.
Run with: pytest tests/test_ci_gate.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.ci_gate import check_thresholds, CI_THRESHOLDS


# ── Fixtures ───────────────────────────────────────────────────────────

def _passing_aggregate():
    """Aggregate dict that passes all CI thresholds."""
    return {
        "retrieval_recall":       0.75,
        "retrieval_precision":    0.65,
        "citation_coverage":      0.80,
        "faithfulness_score":     0.85,
        "hallucination_rate":     0.05,
        "answer_accuracy":        0.70,
        "unanswerable_accuracy":  0.90,
        "mean_latency_ms":        1200.0,
    }


def _failing_aggregate():
    """Aggregate dict that fails every threshold."""
    return {
        "retrieval_recall":       0.30,
        "retrieval_precision":    0.20,
        "citation_coverage":      0.25,
        "faithfulness_score":     0.40,
        "hallucination_rate":     0.50,
        "answer_accuracy":        0.20,
        "unanswerable_accuracy":  0.30,
        "mean_latency_ms":        9000.0,
    }


# ── Tests ──────────────────────────────────────────────────────────────

class TestCheckThresholds:
    def test_all_passing(self):
        passed, failures = check_thresholds(_passing_aggregate())
        assert passed is True
        assert failures == []

    def test_all_failing(self):
        passed, failures = check_thresholds(_failing_aggregate())
        assert passed is False
        # Every threshold except latency (not in CI_THRESHOLDS) should fail
        assert len(failures) > 0

    def test_single_failure_detected(self):
        agg = _passing_aggregate()
        agg["faithfulness_score"] = 0.50  # below 0.70 threshold
        passed, failures = check_thresholds(agg)
        assert passed is False
        failed_keys = [f["metric"] for f in failures]
        assert "faithfulness_score" in failed_keys

    def test_hallucination_rate_upper_bound(self):
        """hallucination_rate uses <= not >= — verify inversion logic."""
        agg = _passing_aggregate()
        agg["hallucination_rate"] = 0.20  # above 0.15 ceiling
        passed, failures = check_thresholds(agg)
        assert passed is False
        failed_keys = [f["metric"] for f in failures]
        assert "hallucination_rate" in failed_keys

    def test_hallucination_rate_at_threshold_passes(self):
        agg = _passing_aggregate()
        agg["hallucination_rate"] = 0.15  # exactly at threshold — should pass (<=)
        passed, failures = check_thresholds(agg)
        assert passed is True

    def test_exactly_at_lower_threshold_passes(self):
        """Metrics at exactly the threshold value must pass."""
        agg = _passing_aggregate()
        agg["retrieval_recall"] = 0.60  # exactly at threshold
        passed, failures = check_thresholds(agg)
        assert "retrieval_recall" not in [f["metric"] for f in failures]

    def test_just_below_lower_threshold_fails(self):
        agg = _passing_aggregate()
        agg["retrieval_recall"] = 0.599
        passed, failures = check_thresholds(agg)
        assert passed is False
        assert "retrieval_recall" in [f["metric"] for f in failures]

    def test_missing_metric_is_skipped(self):
        """Missing metrics produce a warning but don't crash."""
        agg = _passing_aggregate()
        del agg["citation_coverage"]
        # Should not raise
        passed, failures = check_thresholds(agg)
        failed_keys = [f["metric"] for f in failures]
        assert "citation_coverage" not in failed_keys

    def test_failure_dict_structure(self):
        """Each failure dict has the required keys."""
        agg = _failing_aggregate()
        _, failures = check_thresholds(agg)
        for f in failures:
            assert "metric"    in f
            assert "value"     in f
            assert "op"        in f
            assert "threshold" in f

    def test_multiple_failures_all_reported(self):
        """All failing metrics are reported, not just the first."""
        agg = _passing_aggregate()
        agg["retrieval_recall"]    = 0.10
        agg["faithfulness_score"]  = 0.10
        agg["citation_coverage"]   = 0.10
        _, failures = check_thresholds(agg)
        failed_keys = [f["metric"] for f in failures]
        assert "retrieval_recall"   in failed_keys
        assert "faithfulness_score" in failed_keys
        assert "citation_coverage"  in failed_keys

    def test_thresholds_dict_not_empty(self):
        """Sanity check: CI_THRESHOLDS must define at least 5 metrics."""
        assert len(CI_THRESHOLDS) >= 5

    def test_all_thresholds_have_valid_operators(self):
        for metric, (op, threshold) in CI_THRESHOLDS.items():
            assert op in (">=", "<=", ">", "<"), f"Invalid op '{op}' for {metric}"
            assert isinstance(threshold, (int, float)), f"Non-numeric threshold for {metric}"
