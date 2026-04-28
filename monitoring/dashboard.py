"""
monitoring/dashboard.py
Rich terminal dashboard for the monitoring system.

Displays:
  - Live summary: P50/P95 latency, cost, citation coverage, failure rate
  - Latency trend table (hourly buckets)
  - Quality trend table (citation coverage + failure rate over time)
  - Cost trend table
  - Recent anomaly detection (spikes in latency or quality drops)

Usage:
  from monitoring.dashboard import Dashboard
  dash = Dashboard()
  dash.show(window_hours=24)

CLI:
  python cli.py dashboard                  # last 24 hours
  python cli.py dashboard --window 6       # last 6 hours
  python cli.py dashboard --window 168     # last 7 days
"""

from datetime import datetime, timezone
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich import box

from monitoring.metrics_store import MetricsStore

console = Console()

# Thresholds for anomaly highlighting
LATENCY_P95_WARN  = 5000   # ms — yellow warning
LATENCY_P95_CRIT  = 10000  # ms — red critical
FAILURE_RATE_WARN = 0.10   # 10% — yellow
FAILURE_RATE_CRIT = 0.25   # 25% — red
CITATION_WARN     = 0.70   # below this → yellow
CITATION_CRIT     = 0.50   # below this → red


def _color_latency(ms: float) -> str:
    if ms >= LATENCY_P95_CRIT:  return f"[bold red]{ms:.0f} ms[/bold red]"
    if ms >= LATENCY_P95_WARN:  return f"[yellow]{ms:.0f} ms[/yellow]"
    return f"[green]{ms:.0f} ms[/green]"


def _color_rate(rate: float, warn: float, crit: float, invert: bool = True) -> str:
    """invert=True means higher is worse (failure_rate). invert=False means lower is worse (citation_coverage)."""
    pct = f"{rate:.1%}"
    if invert:
        if rate >= crit:  return f"[bold red]{pct}[/bold red]"
        if rate >= warn:  return f"[yellow]{pct}[/yellow]"
        return f"[green]{pct}[/green]"
    else:
        if rate <= crit:  return f"[bold red]{pct}[/bold red]"
        if rate <= warn:  return f"[yellow]{pct}[/yellow]"
        return f"[green]{pct}[/green]"


def _color_cost(usd: float) -> str:
    if usd > 0.01:   return f"[yellow]${usd:.6f}[/yellow]"
    return f"[dim]${usd:.6f}[/dim]"


class Dashboard:
    def __init__(self, store: Optional[MetricsStore] = None):
        self.store = store or MetricsStore()

    def show(self, window_hours: float = 24.0) -> None:
        summary = self.store.summary(window_hours=window_hours)
        console.rule(f"[bold cyan]Ask My Docs — Monitoring Dashboard[/bold cyan]")
        console.print(f"[dim]Window: last {window_hours:.0f}h  |  DB: {self.store.db_path}[/dim]\n")

        if summary.get("total_queries", 0) == 0:
            console.print(
                Panel(
                    "[yellow]No query data in this window.[/yellow]\n"
                    "Run [bold]python cli.py ask[/bold] to generate traffic, "
                    "then check the dashboard again.",
                    title="No Data",
                    border_style="yellow",
                )
            )
            return

        self._summary_panel(summary)
        self._latency_trend(window_hours)
        self._quality_trend(window_hours)
        self._cost_trend(window_hours)
        self._anomalies(window_hours)

    def _summary_panel(self, summary: dict) -> None:
        # Build two side-by-side panels: Performance and Quality
        perf_lines = [
            f"P50 latency  :  {_color_latency(summary['latency_p50_ms'])}",
            f"P95 latency  :  {_color_latency(summary['latency_p95_ms'])}",
            f"Mean latency :  [dim]{summary['latency_mean_ms']:.0f} ms[/dim]",
            f"Total queries:  [white]{summary['total_queries']}[/white]",
            f"Answered     :  [green]{summary['answered_queries']}[/green]",
        ]
        qual_lines = [
            f"Failure rate    :  {_color_rate(summary['failure_rate'], FAILURE_RATE_WARN, FAILURE_RATE_CRIT)}",
            f"Error rate      :  {_color_rate(summary['error_rate'],   FAILURE_RATE_WARN, FAILURE_RATE_CRIT)}",
            f"Citation cov.   :  {_color_rate(summary['citation_coverage_mean'], CITATION_WARN, CITATION_CRIT, invert=False)}",
            f"Faithfulness    :  {_color_rate(summary['faithfulness_mean'],      CITATION_WARN, CITATION_CRIT, invert=False)}",
        ]
        cost_lines = [
            f"Total cost  :  [cyan]${summary['cost_total_usd']:.6f}[/cyan]",
            f"Cost/query  :  [cyan]${summary['cost_mean_usd']:.6f}[/cyan]",
            f"Tokens in   :  [dim]{summary['total_tokens_in']:,}[/dim]",
            f"Tokens out  :  [dim]{summary['total_tokens_out']:,}[/dim]",
        ]

        panels = [
            Panel("\n".join(perf_lines), title="[bold]Performance[/bold]", border_style="cyan",  padding=(0, 2)),
            Panel("\n".join(qual_lines), title="[bold]Quality[/bold]",     border_style="green", padding=(0, 2)),
            Panel("\n".join(cost_lines), title="[bold]Cost[/bold]",        border_style="yellow",padding=(0, 2)),
        ]
        console.print(Columns(panels, equal=True))
        console.print()

    def _latency_trend(self, window_hours: float) -> None:
        trend = self.store.latency_trend(window_hours=window_hours, bucket_minutes=max(15, int(window_hours * 2)))
        if not trend:
            return

        t = Table(title="Latency Trend", box=box.SIMPLE_HEAVY, border_style="dim", show_header=True)
        t.add_column("Bucket (UTC)",  style="dim",   width=22)
        t.add_column("Queries",       style="white", width=9,  justify="right")
        t.add_column("P50",           style="white", width=12, justify="right")
        t.add_column("P95",           style="white", width=12, justify="right")

        for row in trend:
            bucket_str = row["bucket_utc"][:16].replace("T", " ")
            t.add_row(
                bucket_str,
                str(row["count"]),
                _color_latency(row["p50_ms"]),
                _color_latency(row["p95_ms"]),
            )
        console.print(t)
        console.print()

    def _quality_trend(self, window_hours: float) -> None:
        trend = self.store.quality_trend(window_hours=window_hours, bucket_minutes=max(15, int(window_hours * 2)))
        if not trend:
            return

        t = Table(title="Quality Trend", box=box.SIMPLE_HEAVY, border_style="dim")
        t.add_column("Bucket (UTC)",      style="dim",   width=22)
        t.add_column("Queries",           style="white", width=9,  justify="right")
        t.add_column("Citation cov.",     style="white", width=14, justify="right")
        t.add_column("Faithfulness",      style="white", width=14, justify="right")
        t.add_column("Failure rate",      style="white", width=13, justify="right")

        for row in trend:
            bucket_str = row["bucket_utc"][:16].replace("T", " ")
            t.add_row(
                bucket_str,
                str(row["count"]),
                _color_rate(row["citation_coverage"],  CITATION_WARN, CITATION_CRIT, invert=False),
                _color_rate(row["faithfulness"],        CITATION_WARN, CITATION_CRIT, invert=False),
                _color_rate(row["failure_rate"],        FAILURE_RATE_WARN, FAILURE_RATE_CRIT),
            )
        console.print(t)
        console.print()

    def _cost_trend(self, window_hours: float) -> None:
        trend = self.store.cost_trend(window_hours=window_hours, bucket_minutes=max(15, int(window_hours * 2)))
        if not trend:
            return

        t = Table(title="Cost Trend", box=box.SIMPLE_HEAVY, border_style="dim")
        t.add_column("Bucket (UTC)",  style="dim",   width=22)
        t.add_column("Queries",       style="white", width=9,  justify="right")
        t.add_column("Bucket cost",   style="white", width=14, justify="right")
        t.add_column("Cumulative",    style="white", width=14, justify="right")
        t.add_column("Mean/query",    style="white", width=14, justify="right")

        for row in trend:
            bucket_str = row["bucket_utc"][:16].replace("T", " ")
            t.add_row(
                bucket_str,
                str(row["count"]),
                _color_cost(row["bucket_cost_usd"]),
                f"[cyan]${row['cumulative_usd']:.6f}[/cyan]",
                _color_cost(row["mean_cost_usd"]),
            )
        console.print(t)
        console.print()

    def _anomalies(self, window_hours: float) -> None:
        """Detect and highlight significant deviations in the trend."""
        lat_trend  = self.store.latency_trend(window_hours=window_hours, bucket_minutes=60)
        qual_trend = self.store.quality_trend(window_hours=window_hours, bucket_minutes=60)

        anomalies = []

        for row in lat_trend:
            if row["p95_ms"] >= LATENCY_P95_CRIT:
                anomalies.append(
                    f"[bold red]LATENCY SPIKE[/bold red] at {row['bucket_utc'][:16]} — "
                    f"P95 = {row['p95_ms']:.0f} ms (threshold: {LATENCY_P95_CRIT} ms)"
                )

        for row in qual_trend:
            if row["citation_coverage"] <= CITATION_CRIT:
                anomalies.append(
                    f"[bold red]CITATION DROP[/bold red] at {row['bucket_utc'][:16]} — "
                    f"coverage = {row['citation_coverage']:.1%} (threshold: {CITATION_CRIT:.0%})"
                )
            if row["failure_rate"] >= FAILURE_RATE_CRIT:
                anomalies.append(
                    f"[bold red]HIGH FAILURE RATE[/bold red] at {row['bucket_utc'][:16]} — "
                    f"rate = {row['failure_rate']:.1%} (threshold: {FAILURE_RATE_CRIT:.0%})"
                )

        if anomalies:
            lines = "\n".join(f"  • {a}" for a in anomalies)
            console.print(
                Panel(lines, title="[bold red]Anomalies Detected[/bold red]", border_style="red")
            )
        else:
            console.print("[dim]No anomalies detected in this window.[/dim]")
        console.print()
