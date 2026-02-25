"""Investigator agent — decomposes a research topic into subtopics.

Uses a **cheap/fast** model because the task (generating an initial list of
subtopics with brief descriptions) is relatively straightforward.
"""

from __future__ import annotations

import logging
from typing import Any

from research_assistant.agents.base import BaseAgent
from research_assistant.models import InvestigatorOutput

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a research investigator. Given a topic, decompose it into {max_subtopics} \
focused, non-overlapping subtopics suitable for in-depth research.

For each subtopic provide:
- A concise, descriptive name (specific enough to guide deep analysis)
- A one-paragraph explanation of why this subtopic matters to the main \
  topic — focus on what makes it important, contested, or consequential, \
  not just what it is
- A relevance score from 0.0 to 1.0

Prioritise subtopics that reveal interesting tensions, open debates, or \
non-obvious dimensions of the topic. Avoid generic decompositions that \
simply list obvious categories — instead, identify the angles that would \
make a research report genuinely insightful.

Order subtopics by relevance (highest first). Aim for breadth: cover \
historical context, current state, technical depth, societal impact, and \
future outlook where applicable.\
"""


class InvestigatorAgent(BaseAgent):
    """Generates 5–7 research subtopics from a user-supplied topic."""

    AGENT_NAME = "investigator"

    def run(self, *, topic: str, max_subtopics: int = 7, **kwargs: Any) -> dict[str, Any]:
        """Research a topic and return identified subtopics."""
        logger.info("Investigator starting — topic: %s", topic)

        system_msg = _SYSTEM_PROMPT.format(max_subtopics=max_subtopics)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Research topic: {topic}"},
        ]

        parsed = self._call_structured_llm(messages, output_schema=InvestigatorOutput)

        logger.info("Investigator found %d subtopics", len(parsed.subtopics))
        return {"subtopics": parsed.subtopics}
