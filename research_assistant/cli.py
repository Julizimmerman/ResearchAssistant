"""CLI entry point and LangGraph execution loop.

Starts an interactive session: greets the user, asks for a topic,
then runs the full research pipeline with human review.
"""

from __future__ import annotations

import argparse
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langgraph.types import Command

from research_assistant import __version__
from research_assistant.cache import setup_cache
from research_assistant.config import Settings
from research_assistant.cost import CostTracker
from research_assistant.graph import build_graph, initialize_agents
from research_assistant.models import FinalReport, Subtopic
from research_assistant.routing import ModelRouter
from research_assistant.ui import (
    collect_human_input,
    console,
    create_spinner,
    display_final_report,
    display_session_complete,
    display_status,
    display_welcome,
    prompt_for_topic,
    write_cost_summary,
)

logger = logging.getLogger(__name__)


# ── Argument parsing ─────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="research-assistant",
        description="Multi-Agent Research Assistant — interactive AI-powered topic research.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run with mock LLM responses (no API key needed)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose/debug logging",
    )
    parser.add_argument(
        "--max-subtopics",
        type=int,
        default=None,
        help="Maximum subtopics for the Investigator (default: 7)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Custom output path for the report (default: auto-generated)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser.parse_args()


# ── Helpers ──────────────────────────────────────────────────────────


def _sanitise_filename(topic: str) -> str:
    """Convert a topic string into a safe filename fragment."""
    slug = re.sub(r"[^\w\s-]", "", topic.lower())
    slug = re.sub(r"[\s_]+", "_", slug).strip("_")
    return slug[:60]


def _default_output_path(topic: str) -> str:
    """Generate a timestamped output filename."""
    slug = _sanitise_filename(topic)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"report_{slug}_{ts}.md"


def _cost_summary_path(report_path: str) -> str:
    """Derive a cost-summary filename from the report path."""
    p = Path(report_path)
    return str(p.with_name(p.stem + "_costs.txt"))


def _configure_logging(verbose: bool) -> None:
    """Set up Python logging separate from Rich console output."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ── Main ─────────────────────────────────────────────────────────────


def main() -> int:
    """Run the research assistant pipeline. Returns exit code."""
    load_dotenv()
    args = parse_args()
    _configure_logging(args.verbose)

    # ── Load settings ────────────────────────────────────────────────
    settings = Settings()
    if args.max_subtopics is not None:
        settings.MAX_SUBTOPICS = args.max_subtopics

    # ── Validate Azure credentials (unless mock mode) ──────────────
    if not args.mock:
        missing: list[str] = []
        if not settings.AZURE_OPENAI_API_KEY:
            missing.append("AZURE_OPENAI_API_KEY")
        if not settings.AZURE_OPENAI_ENDPOINT:
            missing.append("AZURE_OPENAI_ENDPOINT")
        if missing:
            console.print(
                f"[bold red]Error:[/bold red] Missing environment variable(s): "
                f"{', '.join(missing)}.\n"
                "Set them in a .env file or use [cyan]--mock[/cyan] to test without credentials."
            )
            return 1

    # ── Welcome & topic ──────────────────────────────────────────────
    display_welcome()

    if args.mock:
        console.print("[dim](Running in mock mode — no API calls will be made.)[/dim]\n")

    topic = prompt_for_topic()

    # ── Initialise subsystems ────────────────────────────────────────
    cost_tracker = CostTracker()
    router = ModelRouter(settings, cost_tracker, mock_mode=args.mock)
    initialize_agents(router)

    if not args.mock:
        setup_cache(settings.SQLITE_CACHE_PATH)

    compiled_graph, _ = build_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    start_time = time.monotonic()

    # ── Phase 1: Investigation ───────────────────────────────────────
    display_status("Let me look into that for you...")
    latest_state: dict[str, Any] = {}

    with create_spinner("Researching subtopics"):
        for event in compiled_graph.stream(
            {
                "topic": topic,
                "mock_mode": args.mock,
                "subtopics": [],
                "human_decision": None,
                "curated_analyses": [],
                "final_report": None,
                "cost_records": [],
            },
            config=config,
            stream_mode="values",
        ):
            latest_state = event

    num_subtopics = len(latest_state.get("subtopics", []))
    display_status(f"I identified [bold]{num_subtopics}[/bold] subtopics worth exploring.")

    # ── Phase 2: Human Review (interrupt) ────────────────────────────
    snapshot = compiled_graph.get_state(config)

    if snapshot.next:
        # Graph paused at human_review node — extract interrupt payload
        interrupt_payload = snapshot.tasks[0].interrupts[0].value
        subtopics = [
            Subtopic.model_validate(s) for s in interrupt_payload["subtopics"]
        ]

        human_decision = collect_human_input(subtopics)

        # ── Phase 3 & 4: Curation + Report ──────────────────────────
        active_count = len(human_decision.all_active_subtopics)
        display_status(
            f"Analysing {active_count} subtopic(s) in depth and writing your report..."
        )

        with create_spinner("Working on it"):
            for event in compiled_graph.stream(
                Command(resume=human_decision.model_dump()),
                config=config,
                stream_mode="values",
            ):
                latest_state = event

    # ── Display Results ──────────────────────────────────────────────
    report: FinalReport | None = latest_state.get("final_report")
    cost_records = latest_state.get("cost_records", [])
    elapsed = time.monotonic() - start_time

    if report:
        display_final_report(report)

    # ── Save files ───────────────────────────────────────────────────
    report_path: str | None = None
    cost_path: str | None = None

    if report:
        report_path = args.output or _default_output_path(topic)
        md_content = report.raw_markdown if report.raw_markdown else ""
        if md_content:
            Path(report_path).write_text(md_content, encoding="utf-8")
            logger.info("Report written to %s", report_path)
        else:
            report_path = None
            logger.warning("Report has no raw_markdown content; skipping file write")

    # Write cost summary to a separate file (not shown in chat)
    if report_path and cost_records:
        cost_path = _cost_summary_path(report_path)
        write_cost_summary(cost_records, cost_path)
        logger.info("Cost summary written to %s", cost_path)

    display_session_complete(report_path, elapsed)
    return 0
