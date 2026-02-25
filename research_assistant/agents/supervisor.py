"""Supervisor Agent — orchestrates the full research pipeline.

Wraps the LangGraph execution loop (stream → interrupt → resume) and
manages all data passing between the Investigator, Curator, and Reporter
agents.  Makes no LLM calls itself; its job is coordination.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from langgraph.types import Command

from research_assistant.graph import build_graph, initialize_agents
from research_assistant.models import FinalReport, HumanDecision, Subtopic
from research_assistant.routing import ModelRouter

logger = logging.getLogger(__name__)


class SupervisorAgent:
    """Orchestrates the research pipeline across all agents.

    Responsibilities:
    - Initialise agent instances via the shared ModelRouter
    - Build and run the LangGraph state machine
    - Manage the stream → interrupt → resume cycle for human-in-the-loop
    - Log handoff events between agents
    """

    def __init__(self, router: ModelRouter) -> None:
        self._router = router
        logger.info("Supervisor Agent initialised")

    def run(
        self,
        topic: str,
        collect_human_input: Any,
        on_status: Any,
        on_spinner: Any,
    ) -> tuple[FinalReport | None, float]:
        """Execute the full research pipeline for *topic*.

        Parameters
        ----------
        topic:
            The research topic entered by the user.
        collect_human_input:
            Callable ``(subtopics) -> HumanDecision`` — provided by the CLI
            so the Supervisor never imports UI code directly.
        on_status:
            Callable ``(message: str) -> None`` for brief status updates.
        on_spinner:
            Callable ``(message: str)`` returning a context manager for
            animated spinners.

        Returns
        -------
        (final_report, elapsed_seconds)
        """
        initialize_agents(self._router)
        compiled_graph, _ = build_graph()

        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        start_time = time.monotonic()

        logger.info("Supervisor: starting Investigation phase — topic: %s", topic)
        on_status("Let me look into that for you...")
        latest_state: dict[str, Any] = {}

        with on_spinner("Researching subtopics"):
            for event in compiled_graph.stream(
                {
                    "topic": topic,
                    "subtopics": [],
                    "human_decision": None,
                    "curated_analyses": [],
                    "final_report": None,
                },
                config=config,
                stream_mode="values",
            ):
                latest_state = event

        num_subtopics = len(latest_state.get("subtopics", []))
        logger.info(
            "Supervisor: Investigation complete — %d subtopics found", num_subtopics
        )
        on_status(f"I identified [bold]{num_subtopics}[/bold] subtopics worth exploring.")

        snapshot = compiled_graph.get_state(config)

        if snapshot.next:
            logger.info("Supervisor: handing off to Human Review...")
            interrupt_payload = snapshot.tasks[0].interrupts[0].value
            subtopics = [
                Subtopic.model_validate(s) for s in interrupt_payload["subtopics"]
            ]

            human_decision: HumanDecision = collect_human_input(subtopics)

            active_count = len(human_decision.all_active_subtopics)
            on_status(
                f"Analysing {active_count} subtopic(s) in depth and writing your report..."
            )

            with on_spinner("Working on it"):
                for event in compiled_graph.stream(
                    Command(resume=human_decision.model_dump()),
                    config=config,
                    stream_mode="values",
                ):
                    latest_state = event

            logger.info("Supervisor: handing off to Reporter Agent...")

        report: FinalReport | None = latest_state.get("final_report")
        elapsed = time.monotonic() - start_time

        if report:
            logger.info("Supervisor: Reporter Agent finished — pipeline complete")
        else:
            logger.warning("Supervisor: pipeline finished with no report")

        return report, elapsed
