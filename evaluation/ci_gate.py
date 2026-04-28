"""
evaluation/ci_gate.py
CI gate — reads an evaluation report and enforces metric thresholds.

Exits with code 0 (pass) or 1 (fail) so GitHub Actions / GitLab CI
can fail the build when quality drops below acceptable levels.

Thresholds (edit here to tighten/relax):
  retrieval_recall     >= 0.60   (60% of expected keywords found in top chunks)
  retrieval_precision  >= 0.50   (50% of retrieved chunks are relevant)
  citation_coverage    >= 0.60   (60% of answer sentences are cited)
  faithfulness_score   >= 0.70   (70% of answer claims grounded in chunks)
  hallucination_rate   <= 0.15   (at most 15% of answers have hallucination flag)
  answer_accuracy      >= 0.60   (60% of answerable questions answered correctly)
  unanswerable_accuracy>= 0.80   (80% of unanswerable questions correctly abstained)

Usage (in CI):
  python -m evaluation.ci_gate logs/eval_latest.json
  python -m evaluation.ci_gate           ← uses most recent file in logs/

Usage (programmatic):
  from evaluation.ci_gate import check_report
  passed, failures = check_report(report)
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from rich.console import Console
from rich.table import Table
from rich import box

import config

console = Console()

# ── Thresholds (edit to tighten CI) ────────────────────────────────────
CI_THRESHOLDS: Dict[str, Tuple[str, float]] = {
    # metric_key: (comparison, threshold)
    "retrieval_recall":       (">=", 0.60),
    "retrieval_precision":    (">=", 0.50),
    "citation_coverage":      (">=", 0.60),
    "faithfulness_score":     (">=", 0.70),
    "hallucination_rate":     ("<=", 0.15),
    "answer_accuracy":        (">=", 0.60),
    "unanswerable_accuracy":  (">=", 0.80),
}


def _compare(value: float, op: str, threshold: float) -> bool:
    if op == ">=":
        return value >= threshold
    if op == "<=":
        return value <= threshold
    if op == ">":
        return value > threshold
    if op == "<":
        return value < threshold
    return False


def check_thresholds(aggregate: Dict) -> Tuple[bool, List[Dict]]:
    """
    Check each threshold against the aggregate metrics dict.

    Returns:
      (all_passed: bool, failures: List[{metric, value, op, threshold}])
    """
    failures = []
    for metric, (op, threshold) in CI_THRESHOLDS.items():
        value = aggregate.get(metric)
        if value is None:
            console.print(f"[yellow]Warning: metric '{metric}' not found in report[/yellow]")
            continue
        if not _compare(float(value), op, threshold):
            failures.append({
                "metric":    metric,
                "value":     float(value),
                "op":        op,
                "threshold": threshold,
            })
    return len(failures) == 0, failures


def check_report(report_dict: Dict) -> Tuple[bool, List[Dict]]:
    """Check a report dict (already parsed from JSON) against CI thresholds."""
    aggregate = report_dict.get("aggregate", {})
    return check_thresholds(aggregate)


def _find_latest_report() -> Optional[Path]:
    logs_dir = config.BASE_DIR / "logs"
    if not logs_dir.exists():
        return None
    reports = sorted(logs_dir.glob("eval_*.json"), reverse=True)
    return reports[0] if reports else None


def _print_gate_table(aggregate: Dict, failures: List[Dict]) -> None:
    failed_keys = {f["metric"] for f in failures}

    t = Table(title="CI Gate Results", box=box.ROUNDED, border_style="dim")
    t.add_column("Metric",    style="cyan",  min_width=26)
    t.add_column("Threshold", style="dim",   width=12, justify="right")
    t.add_column("Actual",    style="white", width=10, justify="right")
    t.add_column("Result",    style="white", width=8,  justify="center")

    for metric, (op, threshold) in CI_THRESHOLDS.items():
        value = aggregate.get(metric, 0.0)
        is_pct = metric not in ("mean_latency_ms",)
        fmt = lambda v: f"{v:.1%}" if is_pct else f"{v:.0f}"
        passed_str = (
            "[bold green]PASS[/bold green]"
            if metric not in failed_keys
            else "[bold red]FAIL[/bold red]"
        )
        t.add_row(
            metric,
            f"{op} {fmt(threshold)}",
            fmt(float(value)),
            passed_str,
        )
    console.print(t)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    # Find report path
    if argv:
        report_path = Path(argv[0])
    else:
        report_path = _find_latest_report()
        if report_path is None:
            console.print("[red]No evaluation report found.[/red] Run: python evaluate.py first.")
            sys.exit(2)
        console.print(f"[dim]Using latest report: {report_path}[/dim]")

    if not report_path.exists():
        console.print(f"[red]Report file not found:[/red] {report_path}")
        sys.exit(2)

    with open(report_path, encoding="utf-8") as f:
        report_dict = json.load(f)

    aggregate = report_dict.get("aggregate", {})
    console.rule("[bold cyan]CI Gate Check[/bold cyan]")
    console.print(f"[dim]Report: {report_path}[/dim]")
    console.print(f"[dim]Timestamp: {report_dict.get('timestamp', 'unknown')}[/dim]")
    console.print(f"[dim]Samples: {aggregate.get('total_samples', '?')}[/dim]\n")

    passed, failures = check_thresholds(aggregate)
    _print_gate_table(aggregate, failures)

    if passed:
        console.print("\n[bold green]✓ All CI thresholds passed.[/bold green]")
        sys.exit(0)
    else:
        console.print(f"\n[bold red]✗ {len(failures)} threshold(s) failed:[/bold red]")
        for f in failures:
            console.print(
                f"  [red]•[/red] {f['metric']}: "
                f"got {f['value']:.1%}, need {f['op']} {f['threshold']:.1%}"
            )
        sys.exit(1)


if __name__ == "__main__":
    main()
