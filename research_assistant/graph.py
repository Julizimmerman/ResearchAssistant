"""LangGraph state graph — defines the research pipeline and agent registry.

The graph topology itself serves as the **Supervisor Agent**: it encodes
the linear orchestration flow (investigate → human_review → curate → report)
as declarative edges, eliminating the need for a separate supervisor node.

Node functions are thin wrappers that bridge LangGraph's
``(state) -> partial_state`` contract with the agent classes.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from research_assistant.agents.curator import CuratorAgent
from research_assistant.agents.investigator import InvestigatorAgent
from research_assistant.agents.reporter import ReporterAgent
from research_assistant.models import HumanDecision
from research_assistant.routing import ModelRouter
from research_assistant.state import ResearchState

logger = logging.getLogger(__name__)

# ── Agent registry ───────────────────────────────────────────────────
# Populated once via ``initialize_agents()`` before the graph runs.

_agents: dict[str, Any] = {}


def initialize_agents(router: ModelRouter) -> None:
    """Create and register all agent instances.

    Must be called before ``build_graph().stream()`` so the node functions
    can look up pre-configured agents.
    """
    _agents["investigator"] = InvestigatorAgent(router)
    _agents["curator"] = CuratorAgent(router)
    _agents["reporter"] = ReporterAgent(router)
    logger.info("Agent registry initialised with %d agents", len(_agents))


def get_agent(name: str) -> Any:
    """Retrieve a registered agent by name."""
    return _agents[name]


# ── Node functions ───────────────────────────────────────────────────


def investigate_node(state: ResearchState) -> dict[str, Any]:
    """Node: run the Investigator agent to discover subtopics."""
    agent: InvestigatorAgent = get_agent("investigator")
    return agent.run(topic=state["topic"])


def human_review_node(state: ResearchState) -> dict[str, Any]:
    """Node: pause execution for human review of subtopics.

    Uses LangGraph's ``interrupt()`` to checkpoint the graph and surface
    the subtopics to the calling code. The CLI resumes execution by
    passing a ``Command(resume=human_decision_dict)``.
    """
    subtopics = state["subtopics"]

    # Pause and hand subtopics to the caller.
    human_decision_raw: dict[str, Any] = interrupt(
        {
            "type": "human_review",
            "subtopics": [s.model_dump() for s in subtopics],
        }
    )

    decision = HumanDecision.model_validate(human_decision_raw)
    return {"human_decision": decision}


def curate_node(state: ResearchState) -> dict[str, Any]:
    """Node: run the Curator agent on every approved subtopic."""
    agent: CuratorAgent = get_agent("curator")
    return agent.run(
        topic=state["topic"],
        human_decision=state["human_decision"],  # type: ignore[arg-type]
    )


def report_node(state: ResearchState) -> dict[str, Any]:
    """Node: run the Reporter agent to produce the final Markdown report."""
    agent: ReporterAgent = get_agent("reporter")
    return agent.run(
        topic=state["topic"],
        curated_analyses=state["curated_analyses"],
    )


# ── Graph construction ───────────────────────────────────────────────


def build_graph() -> tuple[CompiledStateGraph, MemorySaver]:
    """Construct and compile the research-assistant state graph.

    Returns ``(compiled_graph, checkpointer)`` so the caller can stream
    events and inspect state snapshots.
    """
    checkpointer = MemorySaver()

    graph = StateGraph(ResearchState)

    graph.add_node("investigate", investigate_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("curate", curate_node)
    graph.add_node("report", report_node)

    graph.add_edge(START, "investigate")
    graph.add_edge("investigate", "human_review")
    graph.add_edge("human_review", "curate")
    graph.add_edge("curate", "report")
    graph.add_edge("report", END)

    compiled = graph.compile(checkpointer=checkpointer)
    logger.info("Research graph compiled successfully")
    return compiled, checkpointer
