"""
monitoring/metrics_store.py
Time-series quality metrics store backed by SQLite.

Tracks per-query:
  - latency_ms            (end-to-end pipeline time)
  - input_tokens          (prompt tokens sent to LLM)
  - output_tokens         (completion tokens received)
  - cost_usd              (estimated cost at Gemini 1.5 Flash pricing)
  - citation_coverage     (fraction of answer sentences with [chunk_N])
  - faithfulness_score    (word-overlap faithfulness proxy)
  - was_answered          (boolean — did the system produce an answer?)
  - retrieval_count       (chunks retrieved before reranking)
  - approved_count        (chunks after citation gate)
  - error                 (error message if pipeline failed)

Aggregation methods:
  - p50/p95 latency
  - cost per request (mean, total)
  - citation coverage rate
  - failure rate (fraction of queries that produced no answer or errored)
  - rolling window summaries (last 1h, 6h, 24h, 7d)

Usage:
  from monitoring.metrics_store import MetricsStore
  store = MetricsStore()
  store.record(query_metric)
  report = store.summary(window_hours=24)
"""

import sqlite3
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List, Any

import config

DB_PATH = config.BASE_DIR / "logs" / "metrics.db"

# Gemini 1.5 Flash pricing (as of 2024-Q4, per 1M tokens)
# Input:  $0.075 / 1M tokens  (prompts ≤128K)
# Output: $0.30  / 1M tokens
GEMINI_FLASH_INPUT_COST_PER_TOKEN  = 0.075  / 1_000_000
GEMINI_FLASH_OUTPUT_COST_PER_TOKEN = 0.30   / 1_000_000


@dataclass
class QueryMetric:
    """One row in the metrics store — one per pipeline query."""
    question:           str
    session_id:         str         = ""
    latency_ms:         float       = 0.0
    retrieval_latency_ms: float     = 0.0
    reranking_latency_ms: float     = 0.0
    generation_latency_ms: float    = 0.0
    input_tokens:       int         = 0
    output_tokens:      int         = 0
    cost_usd:           float       = 0.0
    citation_coverage:  float       = 0.0
    faithfulness_score: float       = 0.0
    was_answered:       bool        = True
    retrieval_count:    int         = 0
    approved_count:     int         = 0
    prompt_version:     str         = "answer_v1"
    model:              str         = "gemini-1.5-flash"
    error:              Optional[str] = None
    ts:                 float       = field(default_factory=time.time)

    @classmethod
    def compute_cost(cls, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens  * GEMINI_FLASH_INPUT_COST_PER_TOKEN +
            output_tokens * GEMINI_FLASH_OUTPUT_COST_PER_TOKEN
        )


class MetricsStore:
    """
    SQLite-backed store for per-query metrics.
    Thread-safe for single-process use.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS query_metrics (
                    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts                    REAL    NOT NULL,
                    question              TEXT    NOT NULL,
                    session_id            TEXT    DEFAULT '',
                    latency_ms            REAL    DEFAULT 0,
                    retrieval_latency_ms  REAL    DEFAULT 0,
                    reranking_latency_ms  REAL    DEFAULT 0,
                    generation_latency_ms REAL    DEFAULT 0,
                    input_tokens          INTEGER DEFAULT 0,
                    output_tokens         INTEGER DEFAULT 0,
                    cost_usd              REAL    DEFAULT 0,
                    citation_coverage     REAL    DEFAULT 0,
                    faithfulness_score    REAL    DEFAULT 0,
                    was_answered          INTEGER DEFAULT 1,
                    retrieval_count       INTEGER DEFAULT 0,
                    approved_count        INTEGER DEFAULT 0,
                    prompt_version        TEXT    DEFAULT 'answer_v1',
                    model                 TEXT    DEFAULT 'gemini-1.5-flash',
                    error                 TEXT    DEFAULT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ts ON query_metrics(ts)"
            )
            conn.commit()

    # ── Write ──────────────────────────────────────────────────────────

    def record(self, metric: QueryMetric) -> None:
        """Insert one QueryMetric row. Never raises — logs silently on error."""
        try:
            with self._conn() as conn:
                conn.execute("""
                    INSERT INTO query_metrics (
                        ts, question, session_id,
                        latency_ms, retrieval_latency_ms, reranking_latency_ms,
                        generation_latency_ms,
                        input_tokens, output_tokens, cost_usd,
                        citation_coverage, faithfulness_score,
                        was_answered, retrieval_count, approved_count,
                        prompt_version, model, error
                    ) VALUES (
                        :ts, :question, :session_id,
                        :latency_ms, :retrieval_latency_ms, :reranking_latency_ms,
                        :generation_latency_ms,
                        :input_tokens, :output_tokens, :cost_usd,
                        :citation_coverage, :faithfulness_score,
                        :was_answered, :retrieval_count, :approved_count,
                        :prompt_version, :model, :error
                    )
                """, {**asdict(metric), "was_answered": int(metric.was_answered)})
                conn.commit()
        except Exception:
            pass   # never crash the pipeline

    # ── Read / aggregation ─────────────────────────────────────────────

    def _rows_since(self, hours: float, conn: sqlite3.Connection) -> List[sqlite3.Row]:
        cutoff = time.time() - hours * 3600
        return conn.execute(
            "SELECT * FROM query_metrics WHERE ts >= ? ORDER BY ts ASC",
            (cutoff,)
        ).fetchall()

    @staticmethod
    def _percentile(values: List[float], pct: float) -> float:
        if not values:
            return 0.0
        s = sorted(values)
        k = (len(s) - 1) * pct / 100
        lo, hi = int(k), min(int(k) + 1, len(s) - 1)
        return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 2)

    def summary(self, window_hours: float = 24.0) -> Dict[str, Any]:
        """
        Aggregate metrics for the last `window_hours` hours.

        Returns a dict with:
          latency_p50, latency_p95, latency_mean
          cost_total_usd, cost_mean_usd
          citation_coverage_mean
          faithfulness_mean
          failure_rate           (fraction not answered or errored)
          error_rate             (fraction with non-null error)
          total_queries
          answered_queries
          total_tokens_in, total_tokens_out
          window_hours
        """
        with self._conn() as conn:
            rows = self._rows_since(window_hours, conn)

        if not rows:
            return {
                "window_hours": window_hours,
                "total_queries": 0,
                "message": "No data in this window.",
            }

        latencies      = [r["latency_ms"]         for r in rows]
        costs          = [r["cost_usd"]            for r in rows]
        cit_coverages  = [r["citation_coverage"]   for r in rows]
        faithfulnesses = [r["faithfulness_score"]  for r in rows]
        answered       = [r for r in rows if r["was_answered"]]
        errors         = [r for r in rows if r["error"] is not None]
        total_in       = sum(r["input_tokens"]  for r in rows)
        total_out      = sum(r["output_tokens"] for r in rows)

        def mean(lst): return round(sum(lst) / len(lst), 4) if lst else 0.0

        return {
            "window_hours":          window_hours,
            "total_queries":         len(rows),
            "answered_queries":      len(answered),
            "failure_rate":          round(1 - len(answered) / len(rows), 4),
            "error_rate":            round(len(errors) / len(rows), 4),
            "latency_p50_ms":        self._percentile(latencies, 50),
            "latency_p95_ms":        self._percentile(latencies, 95),
            "latency_mean_ms":       round(mean(latencies), 1),
            "cost_total_usd":        round(sum(costs), 6),
            "cost_mean_usd":         round(mean(costs), 6),
            "citation_coverage_mean":round(mean(cit_coverages), 4),
            "faithfulness_mean":     round(mean(faithfulnesses), 4),
            "total_tokens_in":       total_in,
            "total_tokens_out":      total_out,
        }

    def latency_trend(
        self,
        window_hours: float = 24.0,
        bucket_minutes: int = 60,
    ) -> List[Dict]:
        """
        Return P50/P95 latency bucketed by time for trend analysis.
        Useful for answering "when did latency spike?"
        """
        with self._conn() as conn:
            rows = self._rows_since(window_hours, conn)

        if not rows:
            return []

        bucket_sec = bucket_minutes * 60
        buckets: Dict[int, List[float]] = {}
        for r in rows:
            b = int(r["ts"] // bucket_sec) * bucket_sec
            buckets.setdefault(b, []).append(r["latency_ms"])

        result = []
        for ts_bucket, lats in sorted(buckets.items()):
            result.append({
                "bucket_ts":  ts_bucket,
                "bucket_utc": datetime.fromtimestamp(ts_bucket, tz=timezone.utc).isoformat(),
                "p50_ms":     self._percentile(lats, 50),
                "p95_ms":     self._percentile(lats, 95),
                "count":      len(lats),
            })
        return result

    def cost_trend(
        self,
        window_hours: float = 24.0,
        bucket_minutes: int = 60,
    ) -> List[Dict]:
        """Cumulative cost and per-query mean cost bucketed by time."""
        with self._conn() as conn:
            rows = self._rows_since(window_hours, conn)

        if not rows:
            return []

        bucket_sec = bucket_minutes * 60
        buckets: Dict[int, List[float]] = {}
        for r in rows:
            b = int(r["ts"] // bucket_sec) * bucket_sec
            buckets.setdefault(b, []).append(r["cost_usd"])

        result = []
        cumulative = 0.0
        for ts_bucket, costs in sorted(buckets.items()):
            bucket_total = sum(costs)
            cumulative  += bucket_total
            result.append({
                "bucket_ts":       ts_bucket,
                "bucket_utc":      datetime.fromtimestamp(ts_bucket, tz=timezone.utc).isoformat(),
                "bucket_cost_usd": round(bucket_total, 6),
                "cumulative_usd":  round(cumulative, 6),
                "mean_cost_usd":   round(bucket_total / len(costs), 6),
                "count":           len(costs),
            })
        return result

    def quality_trend(
        self,
        window_hours: float = 24.0,
        bucket_minutes: int = 60,
    ) -> List[Dict]:
        """Citation coverage and failure rate bucketed by time."""
        with self._conn() as conn:
            rows = self._rows_since(window_hours, conn)

        if not rows:
            return []

        bucket_sec = bucket_minutes * 60
        buckets: Dict[int, List] = {}
        for r in rows:
            b = int(r["ts"] // bucket_sec) * bucket_sec
            buckets.setdefault(b, []).append(r)

        result = []
        for ts_bucket, brows in sorted(buckets.items()):
            cit  = [r["citation_coverage"]  for r in brows]
            fait = [r["faithfulness_score"] for r in brows]
            answered = sum(1 for r in brows if r["was_answered"])
            result.append({
                "bucket_ts":          ts_bucket,
                "bucket_utc":         datetime.fromtimestamp(ts_bucket, tz=timezone.utc).isoformat(),
                "citation_coverage":  round(sum(cit)  / len(cit),  4),
                "faithfulness":       round(sum(fait) / len(fait), 4),
                "failure_rate":       round(1 - answered / len(brows), 4),
                "count":              len(brows),
            })
        return result

    def total_rows(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM query_metrics").fetchone()[0]
