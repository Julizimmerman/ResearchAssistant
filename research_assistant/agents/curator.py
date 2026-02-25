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
    CuratorOutput,
    HumanDecision,
    ReviewedSubtopic,
    Subtopic,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a senior research analyst performing deep, substantive analysis on a \
specific subtopic within a broader research project. Your output will be used \
as the primary source material for a final research report, so depth and \
nuance are critical.

Your analysis must go far beyond surface-level summaries. You are expected to \
reason carefully, explore multiple perspectives, identify tensions and \
tradeoffs, and produce the kind of insight that only comes from thorough \
analytical thinking.

Guidelines for each field:

- summary: Write 3–5 sentences that capture the essence of the subtopic. \
  Do not just define it — explain its current state, why it matters now, \
  and what makes it complex or contested.

- key_findings: Each finding must be a full paragraph (3–4 sentences). \
  State the finding, explain the reasoning or mechanism behind it, cite \
  the type of evidence that supports it (e.g. empirical studies, expert \
  consensus, case studies, statistical trends), and note any caveats or \
  conditions under which the finding holds. Aim for 3–5 findings that \
  cover different dimensions of the subtopic.

- pros: Each pro should be 2–3 sentences. Don't just name the strength — \
  explain WHY it is a strength, what evidence supports it, and who \
  benefits from it. Think about economic, social, technical, and ethical \
  dimensions.

- cons: Each con should be 2–3 sentences. Explain the mechanism of the \
  weakness, who is affected, how severe the risk is, and whether it is \
  inherent or mitigable. Avoid vague criticisms.

- connections: For each connection, explain HOW and WHY this subtopic \
  relates to others in the research. What causal links, dependencies, \
  tensions, or synergies exist? A good connection reveals something \
  non-obvious about how the subtopics interact.

- implications: Write a full paragraph (4–6 sentences) exploring the \
  consequences and second-order effects. What does this analysis mean \
  for the broader topic? What decisions or actions does it inform? \
  What could go wrong if these implications are ignored?

- gaps: For each gap, explain why it matters and what filling it would \
  unlock. A good gap statement identifies not just what is missing, but \
  why the absence is consequential for understanding or decision-making.

Think like a senior consultant presenting to an executive audience: every \
claim must be substantiated, every argument must be developed, and the \
analysis must reveal insights that are not immediately obvious from a \
surface reading of the topic.\
"""


class CuratorAgent(BaseAgent):
    """Analyses each human-approved subtopic in depth."""

    AGENT_NAME = "curator"

    def _curate_one(
        self, topic: str, subtopic: ReviewedSubtopic | Subtopic
    ) -> CuratedAnalysis | None:
        """Curate a single subtopic. Returns analysis or None on failure."""
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
            parsed = self._call_structured_llm(messages, output_schema=CuratorOutput)
            return parsed.analysis
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
        """Curate every active subtopic in parallel and return analyses."""
        active = human_decision.all_active_subtopics
        logger.info("Curator starting — %d subtopics to analyse in parallel", len(active))

        analyses: list[CuratedAnalysis] = []

        with ThreadPoolExecutor(max_workers=len(active)) as executor:
            futures = {
                executor.submit(self._curate_one, topic, subtopic): subtopic
                for subtopic in active
            }
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    analyses.append(result)

        logger.info("Curator finished — %d analyses produced", len(analyses))
        return {"curated_analyses": analyses}
