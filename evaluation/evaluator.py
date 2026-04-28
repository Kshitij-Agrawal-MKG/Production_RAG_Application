"""
evaluation/evaluator.py
Offline evaluation runner for the RAG pipeline.

Loads the golden dataset, runs each question through the pipeline,
computes all metrics, writes a structured JSON results file, and
returns an EvaluationReport with aggregate scores.

Usage:
  from evaluation.evaluator import Evaluator
  ev = Evaluator()
  report = ev.run()
  ev.save_report(report, "logs/eval_2024_01_01.json")
"""

import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional, Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich import box

import config
from evaluation.metrics import compute_all_metrics

console = Console()

# ── Data classes ────────────────────────────────────────────────────────

@dataclass
class SingleResult:
    """Metrics + raw output for one golden dataset entry."""
    id:                     str
    question:               str
    category:               str
    difficulty:             str
    answer:                 str
    expected_keywords:      List[str]
    # metrics
    retrieval_recall:       float
    retrieval_precision:    float
    citation_coverage:      float
    faithfulness_score:     float
    hallucination_flag:     bool
    answered_correctly:     bool
    unanswerable_respected: Optional[bool]
    latency_ms:             float
    # optional debug
    num_retrieved:          int = 0
    num_approved:           int = 0
    error:                  Optional[str] = None


@dataclass
class AggregateMetrics:
    """Mean of each metric across all (or filtered) samples."""
    retrieval_recall:        float = 0.0
    retrieval_precision:     float = 0.0
    citation_coverage:       float = 0.0
    faithfulness_score:      float = 0.0
    hallucination_rate:      float = 0.0   # fraction of samples with flag=True
    answer_accuracy:         float = 0.0   # fraction of answerable Q answered correctly
    unanswerable_accuracy:   float = 0.0   # fraction of unanswerables correctly abstained
    mean_latency_ms:         float = 0.0
    total_samples:           int   = 0
    answerable_samples:      int   = 0
    unanswerable_samples:    int   = 0
    error_samples:           int   = 0


@dataclass
class EvaluationReport:
    """Full evaluation report — saved as JSON."""
    timestamp:      str
    pipeline_config: Dict[str, Any]
    aggregate:      AggregateMetrics
    results:        List[SingleResult] = field(default_factory=list)
    passed_ci:      bool = False          # set by CI gate check


# ── Aggregation helper ─────────────────────────────────────────────────

def _aggregate(results: List[SingleResult]) -> AggregateMetrics:
    if not results:
        return AggregateMetrics()

    valid = [r for r in results if r.error is None]
    answerable   = [r for r in valid if r.unanswerable_respected is None]
    unanswerable = [r for r in valid if r.unanswerable_respected is not None]

    def mean(vals):
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    return AggregateMetrics(
        retrieval_recall     = mean([r.retrieval_recall    for r in valid]),
        retrieval_precision  = mean([r.retrieval_precision for r in valid]),
        citation_coverage    = mean([r.citation_coverage   for r in valid]),
        faithfulness_score   = mean([r.faithfulness_score  for r in valid]),
        hallucination_rate   = mean([float(r.hallucination_flag) for r in valid]),
        answer_accuracy      = mean([float(r.answered_correctly) for r in answerable]) if answerable else 0.0,
        unanswerable_accuracy= mean([float(r.unanswerable_respected) for r in unanswerable]) if unanswerable else 0.0,
        mean_latency_ms      = mean([r.latency_ms for r in valid]),
        total_samples        = len(results),
        answerable_samples   = len(answerable),
        unanswerable_samples = len(unanswerable),
        error_samples        = len(results) - len(valid),
    )


# ── Evaluator ──────────────────────────────────────────────────────────

class Evaluator:
    """
    Runs the full golden dataset through the RAG pipeline
    and computes all Phase 3 evaluation metrics.
    """

    def __init__(
        self,
        golden_dataset_path: Optional[Path] = None,
        pipeline=None,  # inject for testing; else lazy-load
    ):
        self.dataset_path = golden_dataset_path or (config.DATA_DIR / "golden_dataset.json")
        self._pipeline = pipeline  # lazy init

    def _get_pipeline(self):
        if self._pipeline is None:
            # Import here to avoid circular imports and allow offline metric tests
            from rag_pipeline import RAGPipeline
            self._pipeline = RAGPipeline()
        return self._pipeline

    def _load_dataset(self) -> List[Dict]:
        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Golden dataset not found: {self.dataset_path}\n"
                "Create data/golden_dataset.json with Q&A pairs first."
            )
        with open(self.dataset_path, encoding="utf-8") as f:
            data = json.load(f)
        console.print(f"[cyan]Loaded {len(data)} golden dataset entries[/cyan]")
        return data

    def run(
        self,
        limit: Optional[int] = None,
        categories: Optional[List[str]] = None,
    ) -> EvaluationReport:
        """
        Run evaluation over the full golden dataset (or a subset).

        Args:
          limit:      max number of entries to evaluate (None = all)
          categories: filter to specific categories e.g. ["factual", "technical"]

        Returns:
          EvaluationReport
        """
        from datetime import datetime, timezone

        dataset = self._load_dataset()

        if categories:
            dataset = [d for d in dataset if d.get("category") in categories]
            console.print(f"[dim]Filtered to categories {categories}: {len(dataset)} entries[/dim]")

        if limit:
            dataset = dataset[:limit]
            console.print(f"[dim]Limited to first {limit} entries[/dim]")

        pipeline = self._get_pipeline()
        results: List[SingleResult] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Evaluating…", total=len(dataset))

            for entry in dataset:
                progress.update(
                    task,
                    description=f"[cyan]{entry['id']} — {entry['question'][:45]}…"
                )

                start = time.time()
                error_msg = None
                answer = ""
                retrieved_chunks = []
                approved_chunks  = []

                try:
                    result = pipeline.query(
                        entry["question"],
                        verbose=False,
                    )
                    answer           = result.answer
                    retrieved_chunks = result.retrieved_chunks
                    approved_chunks  = result.approved_chunks
                except Exception as e:
                    error_msg = str(e)
                    console.print(f"[red]Error on {entry['id']}:[/red] {e}")

                latency_ms = (time.time() - start) * 1000

                if error_msg:
                    sr = SingleResult(
                        id=entry["id"], question=entry["question"],
                        category=entry.get("category", "unknown"),
                        difficulty=entry.get("difficulty", "unknown"),
                        answer="", expected_keywords=entry.get("expected_keywords", []),
                        retrieval_recall=0.0, retrieval_precision=0.0,
                        citation_coverage=0.0, faithfulness_score=0.0,
                        hallucination_flag=False, answered_correctly=False,
                        unanswerable_respected=None,
                        latency_ms=latency_ms, error=error_msg,
                    )
                else:
                    m = compute_all_metrics(
                        question=entry["question"],
                        answer=answer,
                        expected_keywords=entry.get("expected_keywords", []),
                        expected_answer=entry.get("expected_answer"),
                        retrieved_chunks=retrieved_chunks,
                        approved_chunks=approved_chunks,
                        latency_ms=latency_ms,
                    )
                    sr = SingleResult(
                        id=entry["id"], question=entry["question"],
                        category=entry.get("category", "unknown"),
                        difficulty=entry.get("difficulty", "unknown"),
                        answer=answer,
                        expected_keywords=entry.get("expected_keywords", []),
                        num_retrieved=len(retrieved_chunks),
                        num_approved=len(approved_chunks),
                        **m,
                    )

                results.append(sr)
                progress.advance(task)

        aggregate = _aggregate(results)

        # Pipeline config snapshot (for reproducibility)
        pipeline_config = {
            "embedding_model":  config.EMBEDDING_MODEL,
            "reranker_model":   config.RERANKER_MODEL,
            "gemini_model":     config.GEMINI_MODEL,
            "chunk_size":       config.CHUNK_SIZE,
            "chunk_overlap":    config.CHUNK_OVERLAP,
            "vector_top_k":     config.VECTOR_TOP_K,
            "bm25_top_k":       config.BM25_TOP_K,
            "rerank_top_k":     config.RERANK_TOP_K,
            "rrf_k":            config.RRF_K,
            "min_relevance_score": config.MIN_RELEVANCE_SCORE,
        }

        report = EvaluationReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            pipeline_config=pipeline_config,
            aggregate=aggregate,
            results=results,
        )

        return report

    # ── Persistence ──────────────────────────────────────────────────────

    def save_report(self, report: EvaluationReport, path: Optional[Path] = None) -> Path:
        """Save report as a structured JSON file."""
        from datetime import datetime, timezone

        if path is None:
            config.BASE_DIR.joinpath("logs").mkdir(exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            path = config.BASE_DIR / "logs" / f"eval_{ts}.json"

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert dataclasses to dict for JSON serialisation
        def _serialise(obj):
            if hasattr(obj, "__dataclass_fields__"):
                return asdict(obj)
            return str(obj)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2, default=_serialise)

        console.print(f"[green]Report saved:[/green] {path}")
        return path

    # ── Pretty-print ─────────────────────────────────────────────────────

    def print_report(self, report: EvaluationReport) -> None:
        """Render a human-readable summary table to the terminal."""
        agg = report.aggregate
        console.rule("[bold cyan]Evaluation Report[/bold cyan]")

        # Aggregate metrics table
        t = Table(title="Aggregate Metrics", box=box.ROUNDED, border_style="dim")
        t.add_column("Metric",  style="cyan",  min_width=28)
        t.add_column("Score",   style="white", width=10, justify="right")
        t.add_column("Status",  style="white", width=10, justify="center")

        def _status(val: float, threshold: float, invert: bool = False) -> str:
            ok = val >= threshold if not invert else val <= threshold
            return "[bold green]PASS[/bold green]" if ok else "[bold red]FAIL[/bold red]"

        t.add_row("Retrieval recall",       f"{agg.retrieval_recall:.1%}",       _status(agg.retrieval_recall, 0.60))
        t.add_row("Retrieval precision",    f"{agg.retrieval_precision:.1%}",     _status(agg.retrieval_precision, 0.50))
        t.add_row("Citation coverage",      f"{agg.citation_coverage:.1%}",       _status(agg.citation_coverage, 0.60))
        t.add_row("Faithfulness score",     f"{agg.faithfulness_score:.1%}",      _status(agg.faithfulness_score, 0.70))
        t.add_row("Hallucination rate",     f"{agg.hallucination_rate:.1%}",      _status(agg.hallucination_rate, 0.15, invert=True))
        t.add_row("Answer accuracy",        f"{agg.answer_accuracy:.1%}",         _status(agg.answer_accuracy, 0.60))
        t.add_row("Unanswerable accuracy",  f"{agg.unanswerable_accuracy:.1%}",   _status(agg.unanswerable_accuracy, 0.80))
        t.add_row("Mean latency",           f"{agg.mean_latency_ms:.0f} ms",      "[dim]info[/dim]")
        console.print(t)

        # Sample breakdown
        console.print(
            f"\n[dim]Samples: {agg.total_samples} total | "
            f"{agg.answerable_samples} answerable | "
            f"{agg.unanswerable_samples} unanswerable | "
            f"{agg.error_samples} errors[/dim]"
        )

        # CI gate verdict
        gate = "[bold green]✓ CI PASS[/bold green]" if report.passed_ci else "[bold red]✗ CI FAIL[/bold red]"
        console.print(f"\nCI gate: {gate}")

        # Per-category breakdown
        categories = {}
        for r in report.results:
            if r.error:
                continue
            cat = r.category
            if cat not in categories:
                categories[cat] = {"accuracy": [], "faithfulness": [], "count": 0}
            categories[cat]["accuracy"].append(float(r.answered_correctly))
            categories[cat]["faithfulness"].append(r.faithfulness_score)
            categories[cat]["count"] += 1

        if categories:
            ct = Table(title="By Category", box=box.SIMPLE, border_style="dim")
            ct.add_column("Category",    style="cyan", min_width=16)
            ct.add_column("N",           style="dim",  width=5, justify="right")
            ct.add_column("Accuracy",    style="white",width=10, justify="right")
            ct.add_column("Faithfulness",style="white",width=14, justify="right")
            for cat, v in sorted(categories.items()):
                acc   = sum(v["accuracy"])    / len(v["accuracy"])
                faith = sum(v["faithfulness"]) / len(v["faithfulness"])
                ct.add_row(cat, str(v["count"]), f"{acc:.1%}", f"{faith:.1%}")
            console.print(ct)

        # Top failures
        failures = [r for r in report.results if not r.answered_correctly and r.error is None]
        if failures:
            console.print(f"\n[yellow]Failed questions ({len(failures)}):[/yellow]")
            for r in failures[:8]:
                console.print(f"  [dim]{r.id}[/dim] {r.question[:65]}")
