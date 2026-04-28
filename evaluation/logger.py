"""
evaluation/logger.py
Structured JSONL query logger for monitoring and debugging.

Every pipeline query is written as a single JSON line to logs/queries.jsonl.
This file is the source of truth for:
  - Debugging bad answers (what chunks were retrieved?)
  - Continuous improvement (which questions fail repeatedly?)
  - Drift detection (are scores declining over time?)

Format of each log line:
  {
    "ts":           "2024-01-01T12:00:00Z",
    "session_id":   "abc123",
    "question":     "...",
    "answer":       "...",
    "was_answered": true,
    "num_retrieved": 8,
    "num_approved":  4,
    "sources": [{"source": "...", "chunk_index": 0, "rerank_score": 0.72}],
    "top_chunks": [{"source": "...", "text": "...", "rerank_score": 0.72}],
    "latency_ms": 1240.5
  }

Usage:
  from evaluation.logger import QueryLogger
  logger = QueryLogger()
  logger.log(result, latency_ms=1200.0)
  recent = logger.tail(n=20)
  summary = logger.summary()
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Any

import config

LOG_PATH = config.BASE_DIR / "logs" / "queries.jsonl"


class QueryLogger:
    """
    Append-only JSONL logger for RAG pipeline queries.
    Thread-safe for single-process use (appends one line per query).
    """

    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path or LOG_PATH
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._session_id = uuid.uuid4().hex[:8]

    def log(
        self,
        result,           # RAGResult dataclass
        latency_ms: float = 0.0,
        extra: Optional[Dict] = None,
    ) -> None:
        """
        Append a structured log entry for a completed pipeline query.
        Non-blocking — silently catches write errors so logging never breaks inference.
        """
        try:
            entry = {
                "ts":            datetime.now(timezone.utc).isoformat(),
                "session_id":    self._session_id,
                "question":      result.question,
                "answer":        result.answer,
                "was_answered":  result.was_answered,
                "num_retrieved": len(result.retrieved_chunks),
                "num_approved":  len(result.approved_chunks),
                "sources": [
                    {
                        "source":       s.get("source", ""),
                        "chunk_index":  s.get("chunk_index", 0),
                        "rerank_score": round(s.get("rerank_score", 0.0), 4),
                    }
                    for s in result.sources
                ],
                "top_chunks": [
                    {
                        "source":       c.get("source", ""),
                        "chunk_index":  c.get("chunk_index", 0),
                        "rerank_score": round(c.get("rerank_score", 0.0), 4),
                        "text":         c.get("text", "")[:300],  # truncate for log size
                    }
                    for c in result.approved_chunks
                ],
                "latency_ms": round(latency_ms, 1),
            }
            if extra:
                entry.update(extra)

            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        except Exception:
            pass  # logging must never crash the pipeline

    def tail(self, n: int = 20) -> List[Dict]:
        """Return the last n log entries (most recent last)."""
        if not self.log_path.exists():
            return []
        lines = self.log_path.read_text(encoding="utf-8").splitlines()
        return [json.loads(l) for l in lines[-n:] if l.strip()]

    def summary(self) -> Dict[str, Any]:
        """
        Quick summary stats from the full query log.
        Returns: total_queries, answered_rate, mean_latency_ms, top_unanswered
        """
        if not self.log_path.exists():
            return {"total_queries": 0}

        entries = []
        with open(self.log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        if not entries:
            return {"total_queries": 0}

        answered = [e for e in entries if e.get("was_answered", False)]
        latencies = [e["latency_ms"] for e in entries if "latency_ms" in e]

        unanswered = [e for e in entries if not e.get("was_answered", True)]
        top_unanswered = [e["question"] for e in unanswered[-5:]]

        return {
            "total_queries":    len(entries),
            "answered_count":   len(answered),
            "answered_rate":    round(len(answered) / len(entries), 3),
            "mean_latency_ms":  round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
            "unanswered_count": len(unanswered),
            "top_unanswered":   top_unanswered,
            "log_path":         str(self.log_path),
            "log_size_kb":      round(self.log_path.stat().st_size / 1024, 1),
        }

    def clear(self) -> None:
        """Wipe the query log (with no confirmation — call carefully)."""
        if self.log_path.exists():
            self.log_path.unlink()
