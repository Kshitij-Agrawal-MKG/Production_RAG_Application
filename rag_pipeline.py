"""
rag_pipeline.py
Orchestrator — wires together retriever, reranker, and generator.
Fully instrumented with Langfuse tracing and MetricsStore recording.

Flow:
  query
    → HybridRetriever.retrieve()    (BM25 + Vector → RRF)    [traced]
    → Reranker.rerank()             (cross-encoder + gate)    [traced]
    → Generator.generate()          (Gemini + tokens)         [traced]
    → MetricsStore.record()         (latency, cost, quality)
    → QueryLogger.log()             (JSONL append)
    → RAGResult
"""

from dataclasses import dataclass, field
from typing import List, Dict
import time

from rich.console import Console

import config
from retriever import HybridRetriever
from reranker import Reranker
from generator import Generator
from evaluation.logger import QueryLogger
from evaluation.metrics import citation_coverage, faithfulness_score
from monitoring.tracer import PipelineTracer
from monitoring.metrics_store import MetricsStore, QueryMetric

console = Console()


@dataclass
class RAGResult:
    """Structured output from a RAG pipeline query."""
    question:          str
    answer:            str
    sources:           List[Dict]
    retrieved_chunks:  List[Dict] = field(default_factory=list)
    approved_chunks:   List[Dict] = field(default_factory=list)
    was_answered:      bool  = True
    latency_ms:        float = 0.0
    input_tokens:      int   = 0
    output_tokens:     int   = 0
    cost_usd:          float = 0.0
    citation_coverage: float = 0.0
    faithfulness:      float = 0.0


class RAGPipeline:
    """End-to-end RAG pipeline with full observability."""

    def __init__(self, session_id: str = ""):
        console.rule("[bold cyan]Ask My Docs — RAG Pipeline[/bold cyan]")
        self.retriever  = HybridRetriever()
        self.reranker   = Reranker()
        self.generator  = Generator()
        self.logger     = QueryLogger()
        self.tracer     = PipelineTracer()
        self.metrics    = MetricsStore()
        self.session_id = session_id or ""

        status = "[green]enabled[/green]" if self.tracer.enabled else "[yellow]disabled[/yellow] (add LANGFUSE keys to .env)"
        console.print(f"[dim]Langfuse tracing: {status}[/dim]")
        console.print("[bold green]Pipeline ready.[/bold green]\n")

    def query(
        self,
        question: str,
        vector_k: int  = config.VECTOR_TOP_K,
        bm25_k:   int  = config.BM25_TOP_K,
        rerank_k: int  = config.RERANK_TOP_K,
        verbose:  bool = False,
        user_id:  str  = "",
    ) -> RAGResult:
        """Run the full RAG pipeline with tracing, cost tracking, and quality metrics."""
        console.print(f"\n[bold]Query:[/bold] {question}")
        _t0 = time.time()

        # Start Langfuse trace
        trace_ctx = self.tracer.start_trace(
            question=question,
            session_id=self.session_id,
            user_id=user_id,
            tags=["production"],
        )

        # ── 1. Hybrid retrieval ────────────────────────────────────────
        _t_ret = time.time()
        with console.status("[cyan]Retrieving…[/cyan]"):
            retrieved = self.retriever.retrieve(question, vector_k=vector_k, bm25_k=bm25_k)
        retrieval_ms = (time.time() - _t_ret) * 1000

        with trace_ctx.span_retrieval(question, [], [], retrieved):
            pass

        if verbose and retrieved:
            console.print(f"\n[dim]Retrieved {len(retrieved)} candidates:[/dim]")
            for i, c in enumerate(retrieved[:5], 1):
                console.print(
                    f"  {i}. [{c['source'].split('/')[-1]}, chunk {c['chunk_index']}]"
                    f" rrf={c.get('rrf_score', 0):.4f}"
                )

        # ── 2. Reranking ───────────────────────────────────────────────
        _t_rer = time.time()
        with console.status("[cyan]Reranking…[/cyan]"):
            approved = self.reranker.rerank(question, retrieved, top_k=rerank_k)
        reranking_ms = (time.time() - _t_rer) * 1000

        with trace_ctx.span_reranking(question, retrieved, approved):
            pass

        if verbose and approved:
            console.print(f"\n[dim]Top {len(approved)} after reranking:[/dim]")
            for i, c in enumerate(approved, 1):
                console.print(
                    f"  {i}. [{c['source'].split('/')[-1]}, chunk {c['chunk_index']}]"
                    f" score={c.get('rerank_score', 0):.4f}"
                )

        # ── 3. Answer generation ───────────────────────────────────────
        _t_gen = time.time()
        with console.status("[cyan]Generating answer…[/cyan]"):
            answer, sources, prompt, in_tok, out_tok = self.generator.generate(
                question, approved
            )
        generation_ms = (time.time() - _t_gen) * 1000

        trace_ctx.generation(
            prompt=prompt,
            answer=answer,
            input_tokens=in_tok,
            output_tokens=out_tok,
            model=self.generator.model_name,
            prompt_version=self.generator.prompt_version,
        )

        # ── 4. Quality metrics ─────────────────────────────────────────
        cit_cov      = citation_coverage(answer)
        faith        = faithfulness_score(answer, approved)
        was_answered = bool(sources) or ("could not find" not in answer.lower())
        total_ms     = (time.time() - _t0) * 1000
        cost         = QueryMetric.compute_cost(in_tok, out_tok)

        # Attach quality scores to Langfuse trace
        trace_ctx.score("citation_coverage", cit_cov)
        trace_ctx.score("faithfulness",       faith)
        trace_ctx.score("was_answered",        float(was_answered))
        trace_ctx.finish(metadata={
            "retrieval_count": len(retrieved),
            "approved_count":  len(approved),
            "input_tokens":    in_tok,
            "output_tokens":   out_tok,
            "cost_usd":        round(cost, 8),
        })

        # ── 5. Record to MetricsStore ──────────────────────────────────
        self.metrics.record(QueryMetric(
            question=question,
            session_id=self.session_id,
            latency_ms=round(total_ms, 1),
            retrieval_latency_ms=round(retrieval_ms, 1),
            reranking_latency_ms=round(reranking_ms, 1),
            generation_latency_ms=round(generation_ms, 1),
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
            citation_coverage=cit_cov,
            faithfulness_score=faith,
            was_answered=was_answered,
            retrieval_count=len(retrieved),
            approved_count=len(approved),
            prompt_version=self.generator.prompt_version,
            model=self.generator.model_name,
        ))

        # ── 6. JSONL query log ─────────────────────────────────────────
        rag_result = RAGResult(
            question=question,
            answer=answer,
            sources=sources,
            retrieved_chunks=retrieved,
            approved_chunks=approved,
            was_answered=was_answered,
            latency_ms=round(total_ms, 1),
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=round(cost, 8),
            citation_coverage=cit_cov,
            faithfulness=faith,
        )
        self.logger.log(rag_result, latency_ms=total_ms)

        return rag_result
