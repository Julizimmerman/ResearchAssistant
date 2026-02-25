"""Mock LLM for ``--mock`` mode — runs the full pipeline without API calls.

Returns realistic pre-built Pydantic objects that exercise every downstream
code path (human review, curation, reporting) so reviewers can test the
application without an OpenAI API key.
"""

from __future__ import annotations

import re
from typing import Any

from research_assistant.models import (
    CuratedAnalysis,
    CuratorOutput,
    FinalReport,
    InvestigatorOutput,
    KeyFinding,
    ReportSection,
    Subtopic,
)

# ── Static mock data ─────────────────────────────────────────────────

_MOCK_INVESTIGATOR = InvestigatorOutput(
    subtopics=[
        Subtopic(
            id=1,
            name="Historical Evolution",
            description="Trace the historical development and key milestones that shaped the field.",
            relevance_score=0.90,
        ),
        Subtopic(
            id=2,
            name="Current State of the Art",
            description="Survey the present landscape, major players, and recent breakthroughs.",
            relevance_score=0.95,
        ),
        Subtopic(
            id=3,
            name="Technical Foundations",
            description="Examine the core technical principles and methodologies underpinning the topic.",
            relevance_score=0.85,
        ),
        Subtopic(
            id=4,
            name="Ethical and Social Implications",
            description="Analyse the ethical dilemmas, societal impact, and governance challenges.",
            relevance_score=0.80,
        ),
        Subtopic(
            id=5,
            name="Future Directions",
            description="Project emerging trends, anticipated breakthroughs, and open research questions.",
            relevance_score=0.75,
        ),
    ],
    reasoning=(
        "Decomposed into five complementary perspectives covering historical context, "
        "present landscape, technical depth, ethical considerations, and forward-looking trends."
    ),
)


def _mock_curator(subtopic_name: str) -> CuratorOutput:
    """Generate a realistic mock curation for any subtopic name."""
    return CuratorOutput(
        analysis=CuratedAnalysis(
            subtopic_name=subtopic_name,
            summary=(
                f"In-depth analysis of '{subtopic_name}' reveals several important "
                f"dimensions that warrant careful consideration."
            ),
            key_findings=[
                KeyFinding(
                    finding=f"Primary finding regarding {subtopic_name}: significant progress has been made in recent years.",
                    confidence=0.85,
                    evidence="Supported by a growing body of peer-reviewed literature and industry reports.",
                ),
                KeyFinding(
                    finding=f"Secondary finding: challenges in {subtopic_name} remain underexplored.",
                    confidence=0.72,
                    evidence="Gaps identified through systematic review of existing research.",
                ),
                KeyFinding(
                    finding=f"Tertiary finding: cross-disciplinary approaches strengthen {subtopic_name} outcomes.",
                    confidence=0.68,
                    evidence="Case studies from adjacent fields demonstrate transferable insights.",
                ),
            ],
            pros=[
                "Growing institutional support and funding",
                "Increasing public awareness and engagement",
                "Rapid technological enablement",
            ],
            cons=[
                "Uneven global access and equity concerns",
                "Potential for misuse without adequate governance",
                "Knowledge fragmentation across disciplines",
            ],
            connections=[
                "Connects to ethical frameworks discussed in social implications",
                "Technical foundations directly enable future directions",
            ],
            implications=(
                f"The analysis of '{subtopic_name}' suggests that balancing innovation "
                f"with responsible governance will be critical going forward."
            ),
            gaps=[
                f"Long-term longitudinal studies on {subtopic_name} are lacking",
                "Cross-cultural perspectives remain underrepresented",
            ],
        )
    )


def _mock_reporter(topic: str, num_sections: int) -> FinalReport:
    """Generate a realistic mock final report."""
    sections = [
        ReportSection(
            heading="Introduction",
            content=(
                f"This report presents a multi-faceted examination of **{topic}**, "
                f"synthesising insights from {num_sections} key research areas. "
                f"The analysis draws on current literature, expert perspectives, "
                f"and emerging data to provide a comprehensive overview."
            ),
        ),
        ReportSection(
            heading="Key Findings",
            content=(
                "Across all examined subtopics, several cross-cutting themes emerged:\n\n"
                "1. **Rapid advancement** — the field is evolving faster than governance frameworks can adapt.\n"
                "2. **Equity gaps** — benefits are unevenly distributed across regions and populations.\n"
                "3. **Interdisciplinary potential** — the most impactful work bridges multiple domains.\n"
                "4. **Data limitations** — many conclusions rest on incomplete or biased datasets."
            ),
        ),
        ReportSection(
            heading="Detailed Analysis",
            content=(
                "Each subtopic was analysed for key arguments, supporting evidence, "
                "pros and cons, and connections to the broader research question. "
                "The findings reveal a complex landscape where technological capability "
                "often outpaces ethical and regulatory readiness."
            ),
        ),
        ReportSection(
            heading="Recommendations",
            content=(
                "Based on the analysis, we recommend:\n\n"
                "- Investing in longitudinal research to track long-term impacts\n"
                "- Establishing cross-disciplinary working groups\n"
                "- Developing adaptive governance frameworks\n"
                "- Prioritising equitable access in policy design"
            ),
        ),
    ]
    return FinalReport(
        title=f"Research Report: {topic}",
        executive_summary=(
            f"This report synthesises findings across {num_sections} subtopics related to "
            f"'{topic}'. The analysis reveals a dynamic field with significant potential "
            f"tempered by governance, equity, and data challenges."
        ),
        sections=sections,
        conclusion=(
            f"The multi-agent research pipeline has produced a comprehensive overview of "
            f"'{topic}'. While significant progress is evident, key challenges around "
            f"equity, governance, and interdisciplinary collaboration demand sustained attention."
        ),
        references=[
            "Smith, J. (2024). Advances in the Field: A Comprehensive Review.",
            "Chen, L. & Patel, R. (2024). Cross-Disciplinary Approaches to Complex Problems.",
            "World Economic Forum. (2024). Global Trends Report.",
            "National Academy of Sciences. (2023). Responsible Innovation Framework.",
        ],
        methodology_note=(
            "This report was generated using a multi-agent AI research pipeline. "
            "An Investigator agent identified subtopics, a human reviewer validated them, "
            "a Curator agent performed deep analysis, and a Reporter agent synthesised "
            "the final document. All AI outputs were structured using Pydantic models."
        ),
    )


# ── Mock LLM class ───────────────────────────────────────────────────

# Fake token-usage metadata so cost tracking works in mock mode.
_MOCK_USAGE = {
    "prompt_tokens": 250,
    "completion_tokens": 500,
    "total_tokens": 750,
}


class MockStructuredLLM:
    """Drop-in replacement for ``ChatOpenAI.with_structured_output(...)``.

    Returns pre-built Pydantic objects and includes fake token metadata
    so the cost-tracking pipeline functions identically in mock mode.
    """

    def __init__(self, agent_name: str) -> None:
        self._agent_name = agent_name

    def invoke(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        """Return ``{"raw": AIMessage-like, "parsed": PydanticModel}``."""
        parsed = self._build_response(messages)
        raw = _FakeAIMessage(usage_metadata=_MOCK_USAGE)
        return {"raw": raw, "parsed": parsed}

    def with_structured_output(self, schema: type, **kwargs: Any) -> "MockStructuredLLM":
        """No-op — the mock already returns structured data."""
        return self

    # ── Internal ─────────────────────────────────────────────────────

    def _build_response(self, messages: list[dict[str, str]]) -> Any:
        if self._agent_name == "investigator":
            return _MOCK_INVESTIGATOR

        if self._agent_name == "curator":
            user_msg = messages[-1]["content"] if messages else ""
            match = re.search(r"Subtopic to analyse:\s*(.+?)(?:\n|$)", user_msg)
            if not match:
                match = re.search(r"Subtopic to analyze:\s*(.+?)(?:\n|$)", user_msg)
            name = match.group(1).strip() if match else "Unknown Subtopic"
            return _mock_curator(name)

        if self._agent_name == "reporter":
            user_msg = messages[-1]["content"] if messages else ""
            match = re.search(r"Research topic:\s*(.+?)(?:\n|$)", user_msg)
            topic = match.group(1).strip() if match else "Unknown Topic"
            return _mock_reporter(topic, 3)

        return _MOCK_INVESTIGATOR  # fallback


class _FakeAIMessage:
    """Minimal stand-in for ``langchain_core.messages.AIMessage``."""

    def __init__(self, usage_metadata: dict[str, int]) -> None:
        self.usage_metadata = usage_metadata
        self.response_metadata = {"token_usage": usage_metadata}


# ── Factory ──────────────────────────────────────────────────────────


def create_mock_llm(agent_name: str) -> MockStructuredLLM:
    """Create a mock LLM instance for the given agent."""
    return MockStructuredLLM(agent_name)
