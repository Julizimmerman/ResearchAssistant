"""Curator agent — performs deep analysis on each approved subtopic.

Uses an **expensive/powerful** model because the task requires nuanced
reasoning: synthesising key arguments, pros/cons, cross-subtopic connections,
and identifying knowledge gaps.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from research_assistant.agents.base import BaseAgent
from research_assistant.models import (
    CuratedAnalysis,
    CostRecord,
    CuratorOutput,
    HumanDecision,
    ReviewedSubtopic,
    Subtopic,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a research data curator. Your job is to extract and organise the most \
important FACTS and EVIDENCE about a subtopic — NOT to write prose. The output \
will be used as raw material by a separate agent who will write the actual report.

Provide compact, structured output:
- summary: One concise sentence stating the core finding (max 30 words)
- key_findings: Specific facts, data points, statistics, or conclusions — each \
  as a short statement with a confidence level and the type of evidence behind it
- pros: Short bullet-point strengths or supporting arguments (no prose)
- cons: Short bullet-point weaknesses or counter-arguments (no prose)
- connections: How this subtopic links to adjacent subtopics (keywords only)
- implications: One sentence on what this means for the broader topic
- gaps: Specific unanswered questions or missing evidence

Be terse, factual, and data-driven. Avoid narrative sentences.\
"""


class CuratorAgent(BaseAgent):
    """Analyses each human-approved subtopic in depth."""

    AGENT_NAME = "curator"

    def _curate_one(
        self, topic: str, subtopic: ReviewedSubtopic | Subtopic
    ) -> tuple[CuratedAnalysis, CostRecord] | None:
        """Curate a single subtopic. Returns (analysis, cost) or None on failure."""
        name = subtopic.name
        description = getattr(subtopic, "description", name)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Main research topic: {topic}\n"
                    f"Subtopic to analyse: {name}\n"
                    f"Description: {description}"
                ),
            },
        ]
        try:
            parsed, cost_record = self._call_structured_llm(
                messages,
                output_schema=CuratorOutput,
                task_description=f"Curate: {name}",
            )
            return parsed.analysis, cost_record
        except Exception:
            logger.exception("Failed to curate subtopic '%s' — skipping", name)
            return None

    def run(
        self,
        *,
        topic: str,
        human_decision: HumanDecision,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Curate every active subtopic in parallel and return analyses.

        Returns a partial state dict with keys ``curated_analyses`` and
        ``cost_records``.
        """
        active = human_decision.all_active_subtopics
        logger.info("Curator starting — %d subtopics to analyse in parallel", len(active))

        analyses: list[CuratedAnalysis] = []
        cost_records: list[CostRecord] = []

        with ThreadPoolExecutor(max_workers=len(active)) as executor:
            futures = {
                executor.submit(self._curate_one, topic, subtopic): subtopic
                for subtopic in active
            }
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    analysis, cost_record = result
                    analyses.append(analysis)
                    cost_records.append(cost_record)

        logger.info("Curator finished — %d analyses produced", len(analyses))
        return {
            "curated_analyses": analyses,
            "cost_records": cost_records,
        }
