"""Cost-aware model routing.

Each agent declares a complexity level, and the ``ModelRouter`` maps that
level to the appropriate Azure OpenAI deployment.  Simple tasks get a
cheap/fast deployment; complex tasks get a powerful/expensive one.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from langchain_openai import AzureChatOpenAI

from research_assistant.config import Settings
from research_assistant.cost import CostTracker


class Complexity(str, Enum):
    """Task complexity level used to select a deployment."""

    LOW = "low"  # → cheap deployment  (e.g. gpt-4o-mini)
    HIGH = "high"  # → powerful deployment (e.g. gpt-4o)


# Default complexity for each agent.
AGENT_COMPLEXITY: dict[str, Complexity] = {
    "investigator": Complexity.LOW,
    "curator": Complexity.HIGH,
    "reporter": Complexity.HIGH,
}


class ModelRouter:
    """Routes agents to the appropriate Azure OpenAI deployment based on task complexity."""

    def __init__(
        self,
        settings: Settings,
        cost_tracker: CostTracker,
        *,
        mock_mode: bool = False,
    ) -> None:
        self._settings = settings
        self._cost_tracker = cost_tracker
        self._mock_mode = mock_mode
        self._deployment_map: dict[Complexity, str] = {
            Complexity.LOW: settings.AZURE_OPENAI_DEPLOYMENT_CHEAP,
            Complexity.HIGH: settings.AZURE_OPENAI_DEPLOYMENT_EXPENSIVE,
        }

    # ── Public API ───────────────────────────────────────────────────

    @property
    def cost_tracker(self) -> CostTracker:
        """Expose the shared cost tracker so agents can record usage."""
        return self._cost_tracker

    def get_model_name(self, agent_name: str) -> str:
        """Return the deployment name assigned to *agent_name*."""
        complexity = AGENT_COMPLEXITY.get(agent_name, Complexity.LOW)
        return self._deployment_map[complexity]

    def get_llm(self, agent_name: str) -> Any:
        """Return an ``AzureChatOpenAI`` (or mock) instance for *agent_name*."""
        if self._mock_mode:
            from research_assistant.mock import create_mock_llm

            return create_mock_llm(agent_name)

        deployment = self.get_model_name(agent_name)
        return AzureChatOpenAI(
            azure_deployment=deployment,
            azure_endpoint=self._settings.AZURE_OPENAI_ENDPOINT,
            api_key=self._settings.AZURE_OPENAI_API_KEY,
            api_version=self._settings.AZURE_OPENAI_API_VERSION,
            temperature=0.3,
        )

    def get_structured_llm(self, agent_name: str, output_schema: type) -> Any:
        """Return an LLM bound to *output_schema* via ``with_structured_output``.

        Uses ``include_raw=True`` so callers receive both the parsed Pydantic
        object and the raw ``AIMessage`` (needed for token-usage metadata).
        """
        if self._mock_mode:
            from research_assistant.mock import create_mock_llm

            return create_mock_llm(agent_name)

        llm = self.get_llm(agent_name)
        return llm.with_structured_output(output_schema, include_raw=True)
