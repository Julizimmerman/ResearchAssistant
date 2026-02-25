"""Rich console UI for the research assistant.

All user-facing output goes through this module — no bare ``print()``
calls anywhere in the codebase.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Generator

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from research_assistant.models import (
    CostRecord,
    FinalReport,
    HumanDecision,
    ReviewedSubtopic,
    Subtopic,
    SubtopicAction,
)
from research_assistant.parser import (
    ParseError,
    parse_command_line,
    validate_ids,
)

logger = logging.getLogger(__name__)

console = Console()

# ── Welcome / Banner ─────────────────────────────────────────────────


def display_welcome() -> None:
    """Show a friendly welcome banner and greet the user."""
    console.print()
    console.print(
        Panel(
            "[bold]Welcome![/bold]\n\n"
            "I'm your Research Assistant. I'll help you explore any topic in depth\n"
            "by breaking it into subtopics, letting you pick the ones that matter,\n"
            "and producing a polished report.",
            title="[bold blue]Research Assistant[/bold blue]",
            border_style="blue",
            padding=(1, 2),
        )
    )
    console.print()


def prompt_for_topic() -> str:
    """Ask the user to type a research topic. Loops until valid."""
    while True:
        topic = Prompt.ask(
            "[bold green]What topic would you like to research?[/bold green]"
        ).strip()
        error = validate_topic(topic)
        if error:
            console.print(f"[red]{error}[/red]\n")
            continue
        word_count = len(topic.split())
        if word_count <= 2:
            console.print(
                f"[yellow]Tip:[/yellow] '{topic}' is quite broad — "
                "a more specific phrase usually gives better results.\n"
            )
        return topic


# ── Progress / Spinners ─────────────────────────────────────────────


def create_spinner(message: str) -> Any:
    """Return a ``console.status`` context manager for a spinner."""
    return console.status(f"[bold green]{message}[/bold green]", spinner="dots")


def display_status(message: str) -> None:
    """Print a brief, friendly status update."""
    console.print(f"\n{message}\n")


# ── Subtopic Review ──────────────────────────────────────────────────


def _render_subtopics_table(
    subtopics: list[Subtopic],
    dispositions: dict[int, SubtopicAction] | None = None,
    added: list[Subtopic] | None = None,
) -> None:
    """Render subtopics in a Rich table with optional status column."""
    table = Table(
        show_header=True,
        header_style="bold magenta",
        border_style="dim",
    )
    table.add_column("#", style="cyan", width=4, justify="right")
    table.add_column("Subtopic", style="bold", min_width=20)
    table.add_column("Why it matters", max_width=55)
    table.add_column("Relevance", justify="right", style="green", width=10)

    if dispositions is not None:
        table.add_column("Status", justify="center", width=10)

    for s in subtopics:
        row: list[str] = [
            str(s.id),
            s.name,
            s.description,
            f"{s.relevance_score:.0%}",
        ]
        if dispositions is not None:
            action = dispositions.get(s.id)
            if action == SubtopicAction.APPROVED:
                row.append("[green]approved[/green]")
            elif action == SubtopicAction.REJECTED:
                row.append("[red]rejected[/red]")
            elif action == SubtopicAction.MODIFIED:
                row.append("[yellow]modified[/yellow]")
            else:
                row.append("[dim]pending[/dim]")
        table.add_row(*row)

    # Show user-added subtopics in the same table
    if added:
        for s in added:
            row = [str(s.id), f"[green]{s.name}[/green]", s.description, "—"]
            if dispositions is not None:
                row.append("[green]added[/green]")
            table.add_row(*row)

    console.print()
    console.print(table)
    console.print()


def _display_commands_help() -> None:
    """Show the available review commands (shown once at the start)."""
    console.print(
        Panel(
            "[bold]Commands:[/bold]\n"
            "  [cyan]approve 1,3,5[/cyan]            — approve specific subtopics\n"
            "  [cyan]approve all[/cyan]               — approve everything\n"
            "  [cyan]reject 2,4[/cyan]                — reject specific subtopics\n"
            "  [cyan]add 'Topic Name'[/cyan]          — add a new subtopic\n"
            "  [cyan]modify 3 to 'New Name'[/cyan]    — rename a subtopic\n"
            "  [cyan]done[/cyan]                       — finish and proceed\n\n"
            "  Separate multiple commands with [bold];[/bold]",
            title="[bold yellow]Review Commands[/bold yellow]",
            border_style="yellow",
            padding=(0, 2),
        )
    )
    console.print()


# ── Human-input collection loop ─────────────────────────────────────


def collect_human_input(subtopics: list[Subtopic]) -> HumanDecision:
    """Interactive multi-round review loop.

    Shows subtopics and commands once, then on subsequent rounds only
    shows a compact summary and asks for confirmation.

    Returns a fully-formed ``HumanDecision``.
    """
    console.print(
        "\nHere's what I found. Please review the subtopics below and tell me "
        "which ones to keep, remove, or change.\n"
    )
    _render_subtopics_table(subtopics)
    _display_commands_help()

    valid_ids = {s.id for s in subtopics}
    dispositions: dict[int, SubtopicAction] = {}
    modifications: dict[int, str] = {}  # id → new name
    added: list[Subtopic] = []
    next_id = max(s.id for s in subtopics) + 1
    first_round = True

    while True:
        try:
            line = Prompt.ask("[bold yellow]>[/bold yellow]")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Type 'done' to proceed or Ctrl-C again to quit.[/dim]")
            continue

        # Parse
        try:
            commands = parse_command_line(line)
        except ParseError as exc:
            console.print(f"[red]{exc}[/red]")
            continue

        done = False
        for cmd in commands:
            if cmd.action == "done":
                done = True
                break

            if cmd.action == "approve_all":
                for sid in valid_ids:
                    dispositions[sid] = SubtopicAction.APPROVED
                console.print("[green]All subtopics approved.[/green]")

            elif cmd.action == "approve":
                warnings = validate_ids(cmd.targets, valid_ids)
                for w in warnings:
                    console.print(f"[yellow]{w}[/yellow]")
                approved_names = []
                for tid in cmd.targets:
                    if tid in valid_ids:
                        dispositions[tid] = SubtopicAction.APPROVED
                        name = next((s.name for s in subtopics if s.id == tid), f"#{tid}")
                        approved_names.append(name)
                if approved_names:
                    console.print(f"[green]Approved: {', '.join(approved_names)}[/green]")

            elif cmd.action == "reject":
                warnings = validate_ids(cmd.targets, valid_ids)
                for w in warnings:
                    console.print(f"[yellow]{w}[/yellow]")
                rejected_names = []
                for tid in cmd.targets:
                    if tid in valid_ids:
                        dispositions[tid] = SubtopicAction.REJECTED
                        name = next((s.name for s in subtopics if s.id == tid), f"#{tid}")
                        rejected_names.append(name)
                if rejected_names:
                    console.print(f"[red]Rejected: {', '.join(rejected_names)}[/red]")

            elif cmd.action == "modify":
                tid = cmd.targets[0]
                if tid not in valid_ids:
                    console.print(f"[yellow]Subtopic #{tid} does not exist.[/yellow]")
                else:
                    dispositions[tid] = SubtopicAction.MODIFIED
                    modifications[tid] = cmd.value
                    console.print(f"[green]#{tid} renamed to '{cmd.value}'.[/green]")

            elif cmd.action == "add":
                new_sub = Subtopic(
                    id=next_id,
                    name=cmd.value,
                    description=f"Human-added subtopic: {cmd.value}",
                    relevance_score=1.0,
                )
                added.append(new_sub)
                console.print(f"[green]Added #{next_id}: '{cmd.value}'.[/green]")
                next_id += 1

        if done:
            break

        # Compact status after each command round — no full table re-render
        approved = sum(1 for a in dispositions.values() if a == SubtopicAction.APPROVED)
        rejected = sum(1 for a in dispositions.values() if a == SubtopicAction.REJECTED)
        modified = sum(1 for a in dispositions.values() if a == SubtopicAction.MODIFIED)
        pending = len(valid_ids) - len(dispositions)
        status_parts = []
        if approved:
            status_parts.append(f"[green]{approved} approved[/green]")
        if rejected:
            status_parts.append(f"[red]{rejected} rejected[/red]")
        if modified:
            status_parts.append(f"[yellow]{modified} modified[/yellow]")
        if added:
            status_parts.append(f"[green]{len(added)} added[/green]")
        if pending:
            status_parts.append(f"[dim]{pending} pending[/dim]")
        summary = " · ".join(status_parts) if status_parts else "[dim]no changes yet[/dim]"
        console.print(f"\n[dim]Status:[/dim] {summary}")
        console.print("[dim]Enter more commands, or type 'done' to proceed.[/dim]\n")

    # Build HumanDecision
    reviewed: list[ReviewedSubtopic] = []
    for s in subtopics:
        action = dispositions.get(s.id, SubtopicAction.REJECTED)  # default: reject unless explicitly approved
        name = modifications.get(s.id, s.name)
        reviewed.append(
            ReviewedSubtopic(
                original_id=s.id,
                action=action,
                name=name,
                description=s.description,
            )
        )

    decision = HumanDecision(reviewed_subtopics=reviewed, added_subtopics=added)

    # Warn if nothing is approved
    if not decision.all_active_subtopics:
        console.print(
            "\n[bold red]Heads up:[/bold red] no subtopics are selected — "
            "there won't be anything to analyse."
        )
        proceed = Confirm.ask("Continue anyway?", default=False)
        if not proceed:
            console.print()
            return collect_human_input(subtopics)

    # Show confirmation
    active = decision.all_active_subtopics
    names = [s.name for s in active]
    console.print(
        f"\nGreat — proceeding with [bold]{len(active)}[/bold] subtopic(s): "
        + ", ".join(f"[cyan]{n}[/cyan]" for n in names)
        + "\n"
    )
    return decision


# ── Final Report ─────────────────────────────────────────────────────


def display_final_report(report: FinalReport) -> None:
    """Render the final report section-by-section with styled Rich components."""
    def _pad(renderable: object) -> Padding:
        return Padding(renderable, (1, 0, 0, 0))

    def _renderables() -> Generator[object, None, None]:
        yield Rule(style="green")
        yield _pad(Panel(
            Text(report.title, style="bold white", justify="center"),
            border_style="green",
            padding=(1, 4),
        ))
        if report.executive_summary:
            yield _pad(Panel(
                Markdown(report.executive_summary),
                title="[bold cyan]Executive Summary[/bold cyan]",
                border_style="cyan",
                padding=(1, 2),
            ))
        for s in report.sections:
            yield _pad(Rule(f"[bold]{s.heading}[/bold]", style="dim"))
            yield Padding(Markdown(s.content), (0, 2))
        if report.conclusion:
            yield _pad(Panel(
                Markdown(report.conclusion),
                title="[bold yellow]Conclusion[/bold yellow]",
                border_style="yellow",
                padding=(1, 2),
            ))
        if report.references:
            yield _pad(Rule("[dim]References[/dim]", style="dim"))
            yield from (Padding(Text(f"  • {ref}", style="dim"), (0, 0)) for ref in report.references)
        yield _pad(Rule(style="green"))

    console.print(Group(*_renderables()))
    console.print()


def _fallback_markdown(report: FinalReport) -> str:
    """Build Markdown from structured fields when raw_markdown is empty."""
    lines = [
        f"# {report.title}",
        "",
        "## Executive Summary",
        "",
        report.executive_summary,
        "",
    ]
    for section in report.sections:
        lines.extend([f"## {section.heading}", "", section.content, ""])
    lines.extend(["## Conclusion", "", report.conclusion, ""])
    if report.references:
        lines.append("## References")
        lines.append("")
        for ref in report.references:
            lines.append(f"- {ref}")
    return "\n".join(lines)


# ── Cost Summary (written to file, not shown in chat) ────────────────


def write_cost_summary(cost_records: list[CostRecord], file_path: str) -> None:
    """Write a cost summary to a text file (not shown in the console)."""
    if not cost_records:
        return

    lines: list[str] = ["Cost Summary", "=" * 50, ""]
    total_cost = 0.0
    total_tokens = 0

    lines.append(f"{'Agent':<15} {'Model':<20} {'Tokens':>8} {'Cost (USD)':>12}")
    lines.append("-" * 60)

    for r in cost_records:
        lines.append(
            f"{r.agent_name:<15} {r.model_name:<20} {r.total_tokens:>8,} "
            f"${r.estimated_cost_usd:>11.6f}"
        )
        total_cost += r.estimated_cost_usd
        total_tokens += r.total_tokens

    lines.append("-" * 60)
    lines.append(f"{'TOTAL':<15} {'':<20} {total_tokens:>8,} ${total_cost:>11.6f}")
    lines.append("")

    # Per-model breakdown
    model_stats: dict[str, dict[str, int | float]] = {}
    for r in cost_records:
        if r.model_name not in model_stats:
            model_stats[r.model_name] = {"calls": 0, "tokens": 0, "cost": 0.0}
        model_stats[r.model_name]["calls"] += 1
        model_stats[r.model_name]["tokens"] += r.total_tokens
        model_stats[r.model_name]["cost"] += r.estimated_cost_usd

    lines.append("Per-Model Breakdown")
    lines.append("-" * 40)
    for model, stats in model_stats.items():
        lines.append(
            f"  {model}: {stats['calls']} call(s), "
            f"{stats['tokens']:,} tokens, ${stats['cost']:.6f}"
        )

    Path(file_path).write_text("\n".join(lines), encoding="utf-8")


# ── Execution Summary ────────────────────────────────────────────────


def display_session_complete(
    report_path: str | None,
    elapsed_seconds: float,
) -> None:
    """Show a friendly wrap-up message."""
    parts: list[str] = []
    if report_path:
        parts.append(f"Report saved to [bold]{report_path}[/bold]")
    parts.append(f"Completed in {elapsed_seconds:.1f}s")

    console.print(
        Panel(
            "\n".join(parts),
            title="[bold blue]All done![/bold blue]",
            border_style="blue",
            padding=(1, 2),
        )
    )
    console.print()


# ── Topic validation ─────────────────────────────────────────────────


def validate_topic(topic: str) -> str | None:
    """Return an error message if the topic is invalid, else ``None``."""
    stripped = topic.strip()
    if not stripped:
        return "Please enter a topic — it can't be empty."
    if len(stripped) < 3:
        return f"'{stripped}' is too short. Try something more descriptive."
    return None
