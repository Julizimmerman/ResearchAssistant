"""CLI entry point — handles user I/O and delegates pipeline execution to
the SupervisorAgent.
"""

from __future__ import annotations

import argparse
import logging
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

from dotenv import load_dotenv

from research_assistant import __version__
from research_assistant.agents.supervisor import SupervisorAgent
from research_assistant.config import Settings
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
)

logger = logging.getLogger(__name__)


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
        help="Output path for the report (default: auto-generated)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser.parse_args()


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


def _configure_logging(verbose: bool) -> None:
    """Set up Python logging separate from Rich console output."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> int:
    """Run the research assistant pipeline. Returns exit code."""
    load_dotenv()
    args = parse_args()
    _configure_logging(args.verbose)

    # Load settings 
    settings = Settings()
    if args.max_subtopics is not None:
        settings.MAX_SUBTOPICS = args.max_subtopics

    # Validate Azure credentials (unless mock mode)
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

    # Welcome & topic 
    display_welcome()

    if args.mock:
        console.print("[dim](Running in mock mode — no API calls will be made.)[/dim]\n")

    topic = prompt_for_topic()

    # Run pipeline via Supervisor
    router = ModelRouter(settings, mock_mode=args.mock)
    supervisor = SupervisorAgent(router)

    report, elapsed = supervisor.run(
        topic=topic,
        collect_human_input=collect_human_input,
        on_status=display_status,
        on_spinner=create_spinner,
    )

    # Display results
    if report:
        display_final_report(report)

    # Save report to file
    report_path: str | None = None
    if report:
        md_content = report.raw_markdown or ""
        if md_content:
            report_path = args.output or _default_output_path(topic)
            Path(report_path).write_text(md_content, encoding="utf-8")
            logger.info("Report written to %s", report_path)

    display_session_complete(report_path, elapsed)
    return 0
