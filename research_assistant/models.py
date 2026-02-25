"""Pydantic v2 data models for structured inter-agent communication.

Every piece of data that flows between agents is defined here as a typed
Pydantic model.  These models also serve as the JSON Schema for
``ChatOpenAI.with_structured_output()``, constraining LLM responses to
valid, parseable structures.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Subtopic(BaseModel):
    """A research subtopic identified by the Investigator agent."""

    id: int = Field(..., description="Sequential identifier starting from 1")
    name: str = Field(..., description="Short descriptive name of the subtopic")
    description: str = Field(
        ...,
        description="One-paragraph explanation of why this subtopic is relevant",
    )
    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Relevance to the main research topic (0.0–1.0)",
    )


class InvestigatorOutput(BaseModel):
    """Structured output from the Investigator agent."""

    subtopics: list[Subtopic] = Field(
        ..., description="Identified research subtopics"
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation of the research decomposition strategy",
    )


class SubtopicAction(str, Enum):
    """Disposition assigned to a subtopic during human review."""

    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"


class ReviewedSubtopic(BaseModel):
    """A subtopic after human review, carrying its disposition."""

    original_id: int
    action: SubtopicAction
    name: str = Field(..., description="Name (may have been modified by the reviewer)")
    description: str = Field(
        ..., description="Description (may have been modified by the reviewer)"
    )


class HumanDecision(BaseModel):
    """Complete result of the human review step."""

    reviewed_subtopics: list[ReviewedSubtopic] = Field(default_factory=list)
    added_subtopics: list[Subtopic] = Field(default_factory=list)

    @property
    def approved_subtopics(self) -> list[ReviewedSubtopic]:
        """Subtopics that were approved or modified (not rejected)."""
        return [
            s
            for s in self.reviewed_subtopics
            if s.action in (SubtopicAction.APPROVED, SubtopicAction.MODIFIED)
        ]

    @property
    def all_active_subtopics(self) -> list[ReviewedSubtopic | Subtopic]:
        """Every subtopic that should proceed to the Curator."""
        return list(self.approved_subtopics) + list(self.added_subtopics)


class KeyFinding(BaseModel):
    """A single key finding from deep analysis."""

    finding: str = Field(..., description="The finding statement")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence level (0.0–1.0)"
    )
    evidence: str = Field(..., description="Supporting evidence or reasoning")


class CuratedAnalysis(BaseModel):
    """Deep-analysis result for a single subtopic, produced by the Curator."""

    subtopic_name: str
    summary: str = Field(..., description="Executive summary of the analysis")
    key_findings: list[KeyFinding] = Field(..., description="Major findings")
    pros: list[str] = Field(
        default_factory=list, description="Arguments in favour / strengths"
    )
    cons: list[str] = Field(
        default_factory=list, description="Arguments against / weaknesses"
    )
    connections: list[str] = Field(
        default_factory=list,
        description="Connections to other subtopics in this research",
    )
    implications: str = Field(
        ...,
        description="What these findings imply for the main research topic",
    )
    gaps: list[str] = Field(
        default_factory=list,
        description="Knowledge gaps or areas needing further research",
    )


class CuratorOutput(BaseModel):
    """Wrapper around a single curation result (used for structured output)."""

    analysis: CuratedAnalysis



class ReportSection(BaseModel):
    """A single section of the final Markdown report."""

    heading: str = Field(..., description="Section heading (without # prefix)")
    content: str = Field(..., description="Markdown-formatted section body")


class FinalReport(BaseModel):
    """The complete research report produced by the Reporter agent."""

    title: str
    executive_summary: str
    sections: list[ReportSection]
    conclusion: str
    references: list[str] = Field(
        default_factory=list,
        description="Reference sources cited in the report",
    )
    methodology_note: str = Field(
        default="",
        description="Note about the AI-assisted research methodology",
    )
    raw_markdown: str = Field(
        default="",
        description="Complete report as a single Markdown string (built post-LLM)",
    )
