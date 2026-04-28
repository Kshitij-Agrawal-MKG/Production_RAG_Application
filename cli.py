"""
cli.py
Command-line interface for Ask My Docs.

Commands (Phase 1+2 — core):
  python cli.py ingest <path>                   Ingest file or directory
  python cli.py ask "<question>"                Single-shot Q&A
  python cli.py ask                             Interactive Q&A loop
  python cli.py stats                           Index statistics

Commands (Phase 3 — evaluation):
  python cli.py eval                            Full evaluation run
  python cli.py eval --limit 5                  Quick smoke-test
  python cli.py eval --ci                       Exit 1 if thresholds fail
  python cli.py logs                            Query log summary
  python cli.py logs --tail 20                  Last 20 queries
  python cli.py ci-check                        Gate check latest eval report

Commands (Monitoring & Observability):
  python cli.py dashboard                       24-hour metrics dashboard
  python cli.py dashboard --window 6            Custom time window
  python cli.py prompt-version status           Current prompt version health
  python cli.py prompt-version register         Register a new prompt version
  python cli.py prompt-version verify           Check on-disk vs stored hash

  python cli.py clear                           Wipe all indexes
"""

import sys
import argparse
import shutil
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich import box

import config

console = Console()


# ── Shared result printer ───────────────────────────────────────────────

def print_result(result) -> None:
    """Pretty-print a RAGResult with answer, sources, and cost/token info."""
    console.print()

    answer_md = Markdown(result.answer)
    console.print(
        Panel(
            answer_md,
            title="[bold green]Answer[/bold green]",
            border_style="green",
            padding=(1, 2),
        )
    )

    if result.sources:
        table = Table(
            title="[bold]Cited Sources[/bold]",
            box=box.ROUNDED, show_header=True,
            header_style="bold cyan", border_style="dim",
        )
        table.add_column("#",     style="dim",    width=4)
        table.add_column("File",  style="cyan",   min_width=28)
        table.add_column("Chunk", style="white",  width=7)
        table.add_column("Score", style="yellow", width=8)
        for i, src in enumerate(result.sources, 1):
            table.add_row(
                str(i),
                Path(src["source"]).name,
                str(src["chunk_index"]),
                f"{src.get('rerank_score', 0.0):.3f}",
            )
        console.print(table)
    else:
        console.print("[yellow]No citations produced.[/yellow]")

    # Show cost/token info if available
    if getattr(result, "input_tokens", 0) > 0:
        console.print(
            f"\n[dim]Tokens: {result.input_tokens} in / {result.output_tokens} out  "
            f"│  Cost: ${result.cost_usd:.6f}  "
            f"│  Latency: {result.latency_ms:.0f} ms[/dim]"
        )


# ── Phase 1+2 commands ──────────────────────────────────────────────────

def cmd_ingest(args) -> None:
    from ingest import Ingester
    ingester = Ingester()
    path = Path(args.path)
    if path.is_dir():
        ingester.ingest_directory(path)
    elif path.is_file():
        ingester.ingest_file(path)
        ingester.save_bm25_index()
    else:
        console.print(f"[red]Path not found:[/red] {path}")
        sys.exit(1)


def cmd_ask(args) -> None:
    from rag_pipeline import RAGPipeline
    pipeline = RAGPipeline()

    if args.question:
        result = pipeline.query(" ".join(args.question), verbose=args.verbose)
        print_result(result)
    else:
        console.print(
            Panel(
                "[bold cyan]Ask My Docs[/bold cyan] — Interactive Mode\n"
                "[dim]Type your question and press Enter. "
                "Type [bold]exit[/bold] or [bold]quit[/bold] to stop.[/dim]",
                border_style="cyan",
            )
        )
        while True:
            try:
                question = console.input("\n[bold cyan]You:[/bold cyan] ").strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]Goodbye![/dim]")
                break
            if not question:
                continue
            if question.lower() in {"exit", "quit", "q"}:
                console.print("[dim]Goodbye![/dim]")
                break
            result = pipeline.query(question, verbose=args.verbose)
            print_result(result)
        pipeline.tracer.flush()


def cmd_stats(args) -> None:
    from ingest import Ingester
    try:
        ingester = Ingester()
        stats = ingester.collection_stats()
    except Exception as e:
        console.print(f"[red]Error loading indexes:[/red] {e}")
        return

    table = Table(title="Index Statistics", box=box.ROUNDED, border_style="dim")
    table.add_column("Component",  style="cyan")
    table.add_column("Value",      style="green")
    table.add_row("ChromaDB chunks",  str(stats["chroma_chunks"]))
    table.add_row("BM25 chunks",      str(stats["bm25_chunks"]))
    table.add_row("ChromaDB path",    str(config.CHROMA_DIR))
    table.add_row("BM25 path",        str(config.BM25_INDEX_PATH))
    table.add_row("Embedding model",  config.EMBEDDING_MODEL)
    table.add_row("Reranker model",   config.RERANKER_MODEL)
    table.add_row("LLM",              config.GEMINI_MODEL)
    console.print(table)


def cmd_clear(args) -> None:
    console.print("[bold red]WARNING:[/bold red] This will delete ALL indexed data.")
    confirm = console.input("Type [bold]yes[/bold] to confirm: ").strip().lower()
    if confirm != "yes":
        console.print("[dim]Aborted.[/dim]")
        return
    if config.CHROMA_DIR.exists():
        shutil.rmtree(config.CHROMA_DIR)
        console.print(f"[green]Deleted:[/green] {config.CHROMA_DIR}")
    if config.BM25_INDEX_PATH.exists():
        config.BM25_INDEX_PATH.unlink()
        console.print(f"[green]Deleted:[/green] {config.BM25_INDEX_PATH}")
    console.print("[bold green]All indexes cleared.[/bold green]")


# ── Phase 3 — Evaluation commands ──────────────────────────────────────

def cmd_eval(args) -> None:
    from evaluation.evaluator import Evaluator
    from evaluation.ci_gate import check_thresholds
    from monitoring.prompt_versioner import PromptVersioner
    from dataclasses import asdict

    ev     = Evaluator()
    report = ev.run(limit=args.limit or None, categories=args.categories or None)

    passed, _ = check_thresholds(asdict(report.aggregate))
    report.passed_ci = passed

    ev.print_report(report)
    saved_path = ev.save_report(report)
    console.print(f"\n[dim]Full report saved: {saved_path}[/dim]")

    # Record CI result in prompt version manifest
    pv = PromptVersioner()
    pv.ensure_hashes_populated()
    pv.mark_ci_result(
        version=None,   # uses current version
        passed=passed,
        report_path=str(saved_path),
    )
    console.print(f"[dim]Prompt version CI result recorded: {'PASS' if passed else 'FAIL'}[/dim]")

    if not passed and args.ci:
        sys.exit(1)


def cmd_logs(args) -> None:
    from evaluation.logger import QueryLogger
    logger = QueryLogger()

    if args.tail:
        entries = logger.tail(n=args.tail)
        if not entries:
            console.print("[yellow]No log entries found.[/yellow]")
            return
        t = Table(title=f"Last {len(entries)} queries", box=box.ROUNDED, border_style="dim")
        t.add_column("Time",     style="dim",   width=20)
        t.add_column("Question", style="cyan",  min_width=40)
        t.add_column("Answered", style="white", width=10, justify="center")
        t.add_column("Chunks",   style="white", width=8,  justify="right")
        t.add_column("Latency",  style="white", width=10, justify="right")
        for e in entries:
            answered_str = "[green]yes[/green]" if e.get("was_answered") else "[red]no[/red]"
            t.add_row(
                e.get("ts", "")[:19].replace("T", " "),
                e.get("question", "")[:55],
                answered_str,
                str(e.get("num_approved", 0)),
                f"{e.get('latency_ms', 0):.0f} ms",
            )
        console.print(t)
    else:
        summary = logger.summary()
        if summary.get("total_queries", 0) == 0:
            console.print("[yellow]No queries logged yet.[/yellow]")
            return
        t = Table(title="Query Log Summary", box=box.ROUNDED, border_style="dim")
        t.add_column("Metric", style="cyan",  min_width=22)
        t.add_column("Value",  style="white", width=20, justify="right")
        t.add_row("Total queries",  str(summary["total_queries"]))
        t.add_row("Answered",       f"{summary['answered_count']} ({summary['answered_rate']:.1%})")
        t.add_row("Unanswered",     str(summary["unanswered_count"]))
        t.add_row("Mean latency",   f"{summary['mean_latency_ms']:.0f} ms")
        t.add_row("Log size",       f"{summary['log_size_kb']:.1f} KB")
        t.add_row("Log path",       summary["log_path"])
        console.print(t)
        if summary.get("top_unanswered"):
            console.print("\n[yellow]Recent unanswered questions:[/yellow]")
            for q in summary["top_unanswered"]:
                console.print(f"  • {q}")


def cmd_ci_check(args) -> None:
    from evaluation.ci_gate import main as ci_main
    ci_main([args.report] if args.report else [])


# ── Monitoring commands ─────────────────────────────────────────────────

def cmd_dashboard(args) -> None:
    """Show the monitoring dashboard for a given time window."""
    from monitoring.dashboard import Dashboard
    dash = Dashboard()
    dash.show(window_hours=args.window)


def cmd_prompt_version(args) -> None:
    """Manage prompt versions and integrity."""
    from monitoring.prompt_versioner import PromptVersioner

    pv = PromptVersioner()

    if args.pv_command == "status":
        s = pv.status()
        t = Table(title="Prompt Version Status", box=box.ROUNDED, border_style="dim")
        t.add_column("Field",  style="cyan",  min_width=22)
        t.add_column("Value",  style="white", min_width=40)

        integrity_str = (
            "[green]OK[/green]"  if s["integrity_ok"] is True else
            "[red]MISMATCH — file changed without version bump![/red]" if s["integrity_ok"] is False else
            "[yellow]not yet recorded[/yellow]"
        )
        ci_str = (
            "[green]PASS[/green]" if s["ci_passed"] is True else
            "[red]FAIL[/red]"     if s["ci_passed"] is False else
            "[dim]not run[/dim]"
        )

        t.add_row("Current version",  s["current_version"])
        t.add_row("Prompt file",      s["file"])
        t.add_row("Integrity",        integrity_str)
        t.add_row("Stored hash",      s["stored_hash"])
        t.add_row("On-disk hash",     s["actual_hash"])
        t.add_row("Description",      s["description"] or "[dim]none[/dim]")
        t.add_row("Created",          s["created"] or "[dim]unknown[/dim]")
        t.add_row("CI result",        ci_str)
        t.add_row("Eval report",      s["eval_report"] or "[dim]none[/dim]")
        t.add_row("All versions",     ", ".join(s["all_versions"]) or "[dim]none[/dim]")
        console.print(t)

    elif args.pv_command == "register":
        if not args.version:
            console.print("[red]--version is required for register command[/red]")
            sys.exit(1)
        try:
            pv.register_version(
                version=args.version,
                description=args.description or "",
                changelog=args.changelog or "",
                set_current=not args.no_set_current,
            )
            console.print(
                f"[bold green]Registered:[/bold green] {args.version} "
                f"{'(set as current)' if not args.no_set_current else ''}"
            )
        except FileNotFoundError as e:
            console.print(f"[red]{e}[/red]")
            sys.exit(1)

    elif args.pv_command == "verify":
        try:
            pv.check_for_changes()
            console.print("[bold green]✓ Prompt integrity verified — no unregistered changes.[/bold green]")
        except RuntimeError as e:
            console.print(f"[bold red]{e}[/bold red]")
            sys.exit(1)

    else:
        console.print(f"[red]Unknown sub-command: {args.pv_command}[/red]")
        sys.exit(1)


# ── Entry point ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="Ask My Docs — Production RAG CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py ingest data/docs/sample_kb/
  python cli.py ask "What is the HelixDB refund policy?"
  python cli.py ask --verbose "How does authentication work?"
  python cli.py ask                            (interactive loop)
  python cli.py stats

  python cli.py eval                           (full evaluation)
  python cli.py eval --limit 5 --ci            (smoke-test, fail on regression)
  python cli.py logs --tail 20
  python cli.py ci-check

  python cli.py dashboard                      (24-hour monitoring dashboard)
  python cli.py dashboard --window 6           (last 6 hours)
  python cli.py prompt-version status
  python cli.py prompt-version verify
  python cli.py prompt-version register --version answer_v2 --description "Added CoT"
  python cli.py clear
        """,
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ── ingest ──────────────────────────────────────────────────────────
    p_ingest = sub.add_parser("ingest", help="Ingest documents into the index")
    p_ingest.add_argument("path", help="File or directory to ingest")

    # ── ask ─────────────────────────────────────────────────────────────
    p_ask = sub.add_parser("ask", help="Ask a question (or start interactive loop)")
    p_ask.add_argument("question", nargs="*", help="Question text (omit for interactive)")
    p_ask.add_argument("--verbose", "-v", action="store_true", help="Show retrieval details")

    # ── stats ────────────────────────────────────────────────────────────
    sub.add_parser("stats", help="Show index statistics")

    # ── eval ─────────────────────────────────────────────────────────────
    p_eval = sub.add_parser("eval", help="Run offline evaluation against golden dataset")
    p_eval.add_argument("--limit",      type=int,  default=None, help="Max questions to evaluate")
    p_eval.add_argument("--categories", nargs="+", default=None, help="Filter by category")
    p_eval.add_argument("--ci", action="store_true", help="Exit 1 if any threshold fails")

    # ── logs ─────────────────────────────────────────────────────────────
    p_logs = sub.add_parser("logs", help="Show JSONL query log summary or tail")
    p_logs.add_argument("--tail", type=int, default=None, metavar="N",
                        help="Show last N queries as a table")

    # ── ci-check ─────────────────────────────────────────────────────────
    p_ci = sub.add_parser("ci-check", help="Gate check a saved evaluation report")
    p_ci.add_argument("report", nargs="?", default=None,
                      help="Path to eval JSON (default: latest in logs/)")

    # ── dashboard ────────────────────────────────────────────────────────
    p_dash = sub.add_parser("dashboard",
                             help="Show monitoring dashboard (P50/P95, cost, quality trends)")
    p_dash.add_argument("--window", type=float, default=24.0, metavar="HOURS",
                        help="Time window in hours (default: 24)")

    # ── prompt-version ───────────────────────────────────────────────────
    p_pv = sub.add_parser("prompt-version", help="Manage prompt versions and integrity")
    pv_sub = p_pv.add_subparsers(dest="pv_command", required=True)

    pv_sub.add_parser("status",  help="Show current prompt version status")
    pv_sub.add_parser("verify",  help="Verify on-disk prompt matches stored hash")

    p_pv_reg = pv_sub.add_parser("register", help="Register a new prompt version")
    p_pv_reg.add_argument("--version",       required=True, help="Version name e.g. answer_v2")
    p_pv_reg.add_argument("--description",   default="",    help="Short description")
    p_pv_reg.add_argument("--changelog",     default="",    help="What changed vs previous version")
    p_pv_reg.add_argument("--no-set-current",action="store_true",
                           help="Register without making this the active version")

    # ── clear ────────────────────────────────────────────────────────────
    sub.add_parser("clear", help="Delete all indexes (asks for confirmation)")

    args = parser.parse_args()

    dispatch = {
        "ingest":         cmd_ingest,
        "ask":            cmd_ask,
        "stats":          cmd_stats,
        "eval":           cmd_eval,
        "logs":           cmd_logs,
        "ci-check":       cmd_ci_check,
        "dashboard":      cmd_dashboard,
        "prompt-version": cmd_prompt_version,
        "clear":          cmd_clear,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
