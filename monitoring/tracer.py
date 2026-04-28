"""
monitoring/tracer.py
Langfuse distributed tracing for the RAG pipeline.

Every pipeline query becomes a Langfuse trace with nested spans:
  trace: rag_query
    span: retrieval        (BM25 + vector + RRF)
    span: reranking        (cross-encoder scores)
    generation: llm_call   (prompt, completion, token counts)

Degrades gracefully: if Langfuse is not configured (no keys in .env),
all tracing calls are no-ops and the pipeline runs normally.

Setup:
  1. Sign up free at https://cloud.langfuse.com
  2. Create a project and copy your keys
  3. Add to .env:
       LANGFUSE_PUBLIC_KEY=pk-lf-...
       LANGFUSE_SECRET_KEY=sk-lf-...
       LANGFUSE_HOST=https://cloud.langfuse.com   (or your self-hosted URL)

Usage:
  from monitoring.tracer import PipelineTracer
  tracer = PipelineTracer()
  ctx = tracer.start_trace(question, session_id="abc")
  with ctx.span_retrieval(query, [], [], merged):
      pass
  with ctx.span_reranking(query, candidates, approved):
      pass
  ctx.generation(prompt, answer, input_tokens, output_tokens)
  ctx.finish()
"""

import os
import time
from contextlib import contextmanager
from typing import List, Dict, Optional, Any

from dotenv import load_dotenv

load_dotenv()

# ── Langfuse availability check ────────────────────────────────────────
_LANGFUSE_AVAILABLE = False
_langfuse_client = None

_PK  = os.getenv("LANGFUSE_PUBLIC_KEY", "")
_SK  = os.getenv("LANGFUSE_SECRET_KEY", "")
_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

if _PK and _SK:
    try:
        from langfuse import Langfuse
        _langfuse_client = Langfuse(
            public_key=_PK,
            secret_key=_SK,
            host=_HOST,
        )
        _LANGFUSE_AVAILABLE = True
    except ImportError:
        pass   # langfuse not installed — silent no-op mode
    except Exception:
        pass   # bad credentials or network — silent no-op mode


def is_tracing_enabled() -> bool:
    return _LANGFUSE_AVAILABLE


# ── Span wrappers ──────────────────────────────────────────────────────

class _NoOpSpan:
    """Context manager that does nothing when Langfuse is unavailable."""
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def update(self, **kwargs): pass
    def end(self, **kwargs): pass


class _LiveSpan:
    """Wraps a real Langfuse span."""
    def __init__(self, span):
        self._span = span

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        status = "ERROR" if exc_type else "SUCCESS"
        try:
            self._span.end(status_message=status)
        except Exception:
            pass

    def update(self, **kwargs):
        try:
            self._span.update(**kwargs)
        except Exception:
            pass


class TraceContext:
    """
    Holds the active Langfuse trace and exposes span factories for each
    pipeline stage. All methods are safe to call when tracing is disabled.
    """

    def __init__(self, trace, question: str, session_id: str):
        self._trace    = trace
        self.question  = question
        self.session_id = session_id
        self._t0        = time.time()

    # ── Retrieval span ────────────────────────────────────────────────

    @contextmanager
    def span_retrieval(
        self,
        query: str,
        vector_hits: List[Dict],
        bm25_hits: List[Dict],
        merged: List[Dict],
    ):
        """Span covering BM25 + vector search + RRF merge."""
        if not _LANGFUSE_AVAILABLE or self._trace is None:
            yield _NoOpSpan()
            return

        span = None
        try:
            span = self._trace.span(
                name="retrieval",
                input={"query": query},
                metadata={
                    "vector_hits": len(vector_hits),
                    "bm25_hits":   len(bm25_hits),
                },
            )
        except Exception:
            yield _NoOpSpan()
            return

        wrapper = _LiveSpan(span)
        try:
            yield wrapper
            top_merged = [
                {
                    "source":      c.get("source", ""),
                    "chunk_index": c.get("chunk_index", 0),
                    "rrf_score":   round(c.get("rrf_score", 0.0), 4),
                    "text_preview": c.get("text", "")[:120],
                }
                for c in merged[:5]
            ]
            try:
                span.update(
                    output={"merged_count": len(merged), "top_5": top_merged}
                )
            except Exception:
                pass
        finally:
            try:
                span.end()
            except Exception:
                pass

    # ── Reranking span ────────────────────────────────────────────────

    @contextmanager
    def span_reranking(
        self,
        query: str,
        candidates: List[Dict],
        approved: List[Dict],
    ):
        """Span covering cross-encoder scoring and citation gate."""
        if not _LANGFUSE_AVAILABLE or self._trace is None:
            yield _NoOpSpan()
            return

        span = None
        try:
            span = self._trace.span(
                name="reranking",
                input={
                    "query":           query,
                    "candidate_count": len(candidates),
                },
            )
        except Exception:
            yield _NoOpSpan()
            return

        wrapper = _LiveSpan(span)
        try:
            yield wrapper
            scored = [
                {
                    "source":       c.get("source", "").split("/")[-1],
                    "chunk_index":  c.get("chunk_index", 0),
                    "rerank_score": round(c.get("rerank_score", 0.0), 4),
                    "text_preview": c.get("text", "")[:120],
                }
                for c in approved
            ]
            try:
                span.update(
                    output={
                        "approved_count": len(approved),
                        "dropped_count":  len(candidates) - len(approved),
                        "approved_chunks": scored,
                    }
                )
            except Exception:
                pass
        finally:
            try:
                span.end()
            except Exception:
                pass

    # ── LLM generation span ───────────────────────────────────────────

    def generation(
        self,
        prompt:        str,
        answer:        str,
        input_tokens:  int,
        output_tokens: int,
        model:         str = "",
        prompt_version: str = "answer_v1",
    ) -> None:
        """Record the LLM generation event with token counts."""
        if not _LANGFUSE_AVAILABLE or self._trace is None:
            return
        try:
            self._trace.generation(
                name="llm_call",
                model=model or "gemini-1.5-flash",
                model_parameters={
                    "prompt_version": prompt_version,
                },
                input=prompt,
                output=answer,
                usage={
                    "input":  input_tokens,
                    "output": output_tokens,
                    "total":  input_tokens + output_tokens,
                    "unit":   "TOKENS",
                },
            )
        except Exception:
            pass

    # ── Trace-level score ─────────────────────────────────────────────

    def score(self, name: str, value: float, comment: str = "") -> None:
        """Attach a numeric score to the trace (e.g. citation_coverage=0.9)."""
        if not _LANGFUSE_AVAILABLE or self._trace is None:
            return
        try:
            _langfuse_client.score(
                trace_id=self._trace.id,
                name=name,
                value=value,
                comment=comment or None,
            )
        except Exception:
            pass

    def finish(self, metadata: Optional[Dict] = None) -> None:
        """Finalise the trace with end-to-end latency."""
        if not _LANGFUSE_AVAILABLE or self._trace is None:
            return
        try:
            elapsed = round((time.time() - self._t0) * 1000, 1)
            extra = {"latency_ms": elapsed}
            if metadata:
                extra.update(metadata)
            self._trace.update(metadata=extra)
        except Exception:
            pass


# ── PipelineTracer ─────────────────────────────────────────────────────

class PipelineTracer:
    """
    Main entry point for tracing. Produces a TraceContext per query.

    Usage:
        tracer = PipelineTracer()
        ctx = tracer.start_trace(question, session_id="abc")
        # ... run pipeline ...
        ctx.finish()
    """

    def __init__(self):
        self.enabled = _LANGFUSE_AVAILABLE
        if self.enabled:
            self._lf = _langfuse_client
        else:
            self._lf = None

    def start_trace(
        self,
        question: str,
        session_id: str = "",
        user_id: str = "",
        tags: Optional[List[str]] = None,
    ) -> TraceContext:
        """
        Start a new Langfuse trace for a pipeline query.
        Returns a TraceContext — safe to use even if tracing is disabled.
        """
        if not self.enabled:
            return TraceContext(trace=None, question=question, session_id=session_id)

        try:
            trace = self._lf.trace(
                name="rag_query",
                input=question,
                session_id=session_id or None,
                user_id=user_id or None,
                tags=tags or [],
                metadata={"question_length": len(question)},
            )
            return TraceContext(trace=trace, question=question, session_id=session_id)
        except Exception:
            return TraceContext(trace=None, question=question, session_id=session_id)

    def flush(self) -> None:
        """Flush pending events to Langfuse. Call at process exit in batch jobs."""
        if self.enabled and self._lf:
            try:
                self._lf.flush()
            except Exception:
                pass
