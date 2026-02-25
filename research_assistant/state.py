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

    topic: str
    subtopics: list[Subtopic]
    human_decision: HumanDecision | None
    curated_analyses: Annotated[list[CuratedAnalysis], add]
    final_report: FinalReport | None
