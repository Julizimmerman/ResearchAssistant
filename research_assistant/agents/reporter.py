"""Reporter agent — synthesises curated analyses into a Markdown report.

Uses the **most powerful** model available because high-quality writing,
synthesis across multiple subtopics, and a coherent narrative require strong
language capabilities.
"""

from __future__ import annotations

import logging
from typing import Any

from research_assistant.agents.base import BaseAgent
from research_assistant.models import CuratedAnalysis, FinalReport

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a professional research report writer. You will receive structured \
research data (facts, evidence, findings) collected by a curator. Your job is \
to WRITE AN ORIGINAL REPORT — do not copy or paraphrase the curator's text. \
Use the structured data as your source of truth, then write in your own voice.

Rules:
- DO NOT copy curator sentences verbatim. Transform facts into flowing prose.
- Each subtopic section must EXPAND on the curator data: add context, \
  explain significance, connect ideas, and draw insights the data implies.
- The executive summary must synthesise ACROSS all subtopics — not describe \
  each one individually.
- The conclusion must identify cross-cutting themes and tensions between \
  subtopics — not restate each section.
- Use Markdown formatting (headings, bullet points, bold, tables) for \
  readability.

The report MUST include:
1. A compelling, specific title
2. An executive summary (2–3 paragraphs) synthesising the whole
3. One detailed section per subtopic — written as original analysis, not a \
   reformatting of curator notes
4. A conclusion that draws cross-cutting insights
5. References (illustrative if necessary)
6. A brief methodology note\
"""


def _format_analyses_for_prompt(analyses: list[CuratedAnalysis]) -> str:
    """Serialise curated analyses as structured source data for the reporter."""
    parts: list[str] = []
    for a in analyses:
        findings = "\n".join(
            f"  • [{f.confidence:.0%}] {f.finding} (evidence: {f.evidence})"
            for f in a.key_findings
        )
        pros = "\n".join(f"  + {p}" for p in a.pros) or "  (none)"
        cons = "\n".join(f"  - {c}" for c in a.cons) or "  (none)"
        connections = ", ".join(a.connections) or "(none)"
        gaps = "\n".join(f"  ? {g}" for g in a.gaps) or "  (none)"

        parts.append(
            f"[SUBTOPIC: {a.subtopic_name}]\n"
            f"Core finding: {a.summary}\n"
            f"Implication: {a.implications}\n\n"
            f"Evidence/findings:\n{findings}\n\n"
            f"Strengths:\n{pros}\n\n"
            f"Weaknesses:\n{cons}\n\n"
            f"Links to: {connections}\n\n"
            f"Open questions:\n{gaps}"
        )
    return "\n\n════════\n\n".join(parts)


def build_raw_markdown(report: FinalReport) -> str:
    """Assemble the report's structured fields into a single Markdown string."""
    lines: list[str] = [
        f"# {report.title}",
        "",
        "## Executive Summary",
        "",
        report.executive_summary,
        "",
        "## Table of Contents",
        "",
    ]

    # Table of contents
    for i, section in enumerate(report.sections, 1):
        anchor = section.heading.lower().replace(" ", "-")
        lines.append(f"{i}. [{section.heading}](#{anchor})")
    lines.append(f"{len(report.sections) + 1}. [Conclusion](#conclusion)")
    lines.append("")

    # Sections
    for section in report.sections:
        lines.extend([f"## {section.heading}", "", section.content, ""])

    # Conclusion
    lines.extend(["## Conclusion", "", report.conclusion, ""])

    # References
    if report.references:
        lines.append("## References")
        lines.append("")
        for ref in report.references:
            lines.append(f"- {ref}")
        lines.append("")

    # Methodology note
    if report.methodology_note:
        lines.extend(["---", "", f"*{report.methodology_note}*", ""])

    return "\n".join(lines)


class ReporterAgent(BaseAgent):
    """Generates a polished Markdown research report from curated analyses."""

    AGENT_NAME = "reporter"

    def run(
        self,
        *,
        topic: str,
        curated_analyses: list[CuratedAnalysis],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Produce the final report.

        Returns a partial state dict with keys ``final_report`` and
        ``cost_records``.
        """
        logger.info(
            "Reporter starting — synthesising %d analyses", len(curated_analyses)
        )

        analyses_text = _format_analyses_for_prompt(curated_analyses)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Research topic: {topic}\n\n"
                    f"Number of subtopics analysed: {len(curated_analyses)}\n\n"
                    f"Curated analyses:\n\n{analyses_text}"
                ),
            },
        ]

        parsed, cost_record = self._call_structured_llm(
            messages,
            output_schema=FinalReport,
            task_description=f"Report: {topic}",
        )

        # Build the full Markdown from the structured report.
        raw_md = build_raw_markdown(parsed)

        # Attach raw_markdown by reconstructing with the field set.
        report = parsed.model_copy(update={"raw_markdown": raw_md})

        logger.info("Reporter finished — report has %d sections", len(report.sections))
        return {
            "final_report": report,
            "cost_records": [cost_record],
        }
