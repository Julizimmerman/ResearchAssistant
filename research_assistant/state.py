"""LangGraph shared state definition.

The ``ResearchState`` TypedDict flows through every node in the graph.
Fields using ``Annotated[list, add]`` are *reducers*: each node can return
a partial list that gets **appended** to the accumulated state rather than
replacing it.
"""

from __future__ import annotations

from operator import add
from typing import Annotated, TypedDict

from research_assistant.models import (
    CuratedAnalysis,
    FinalReport,
    HumanDecision,
    Subtopic,
)


class ResearchState(TypedDict):
    """Shared state flowing through the research-assistant graph."""

    # ── Input ────────────────────────────────────────────────────────
    topic: str

    # ── Investigator output ──────────────────────────────────────────
    subtopics: list[Subtopic]

    # ── Human-review output ──────────────────────────────────────────
    human_decision: HumanDecision | None

    # ── Curator output (append-reducer) ──────────────────────────────
    curated_analyses: Annotated[list[CuratedAnalysis], add]

    # ── Reporter output ──────────────────────────────────────────────
    final_report: FinalReport | None

    # ── Control ──────────────────────────────────────────────────────
    mock_mode: bool
