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
You are a senior research report writer producing a comprehensive, \
publication-quality report. You will receive structured research data \
(findings, evidence, analysis) collected by a research curator. Your job \
is to transform this raw analytical material into a polished, cohesive \
research document written entirely in your own voice.

CRITICAL RULES:
- NEVER copy or closely paraphrase the curator's text. Use the data as \
  source material, then write original prose.
- Every section must ADD VALUE beyond what the curator provided: add \
  context, explain significance, draw connections the data implies, and \
  surface insights that emerge from looking at the evidence as a whole.

STRUCTURE (all sections required):

1. TITLE: Must be specific and compelling. Not "Research Report on X" but \
   something that captures the key insight or tension discovered. Example: \
   instead of "Research Report: Artificial Intelligence" write something like \
   "The Double-Edged Sword: How AI's Rapid Advancement Outpaces the \
   Frameworks Meant to Govern It".

2. EXECUTIVE SUMMARY (3–4 paragraphs): This is the most important section. \
   It must synthesise ACROSS all subtopics into a unified narrative — not \
   summarise each subtopic sequentially. What is the overarching story? \
   What are the key tensions? What should the reader take away? A busy \
   executive who reads only this section should understand the full picture.

3. SUBTOPIC SECTIONS (one per subtopic, 3–5 paragraphs each): Each section \
   should read as a self-contained analytical essay. Open with context, \
   develop the key arguments with evidence, address counterpoints, and \
   close with implications. Use smooth transitions between sections to \
   build a narrative arc across the report.

4. CONCLUSION (2–3 paragraphs): Identify cross-cutting themes and tensions \
   BETWEEN subtopics — do not restate individual sections. What patterns \
   emerge? What tradeoffs must be navigated? What remains unresolved? \
   End with a forward-looking perspective.

5. REFERENCES: List sources cited or referenced. Use illustrative academic \
   and institutional sources where appropriate.

6. METHODOLOGY NOTE (brief): Explain that this report was produced using a \
   multi-agent AI research pipeline with human-in-the-loop validation, and \
   briefly describe the roles of each agent.

FORMATTING:
- Use Markdown: headings (##), bold for key terms, bullet points only when \
  listing discrete items (not as a substitute for prose).
- Prefer flowing paragraphs over bullet-point lists for analytical content.
- Aim for depth and substance — a complete report should be thorough enough \
  to be genuinely informative to someone unfamiliar with the topic.\
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
        """Produce the final report."""
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

        parsed = self._call_structured_llm(messages, output_schema=FinalReport)

        raw_md = build_raw_markdown(parsed)
        report = parsed.model_copy(update={"raw_markdown": raw_md})

        logger.info("Reporter finished — report has %d sections", len(report.sections))
        return {"final_report": report}
