"""Abstract base class for all research agents."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from research_assistant.routing import ModelRouter

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class providing shared infrastructure for every agent.

    Subclasses must set ``AGENT_NAME`` and implement ``run()``.
    """

    AGENT_NAME: str = ""  # overridden by each subclass

    def __init__(self, router: ModelRouter) -> None:
        self._router = router

    @abstractmethod
    def run(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the agent's task and return a partial state dict."""
        ...

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _call_structured_llm(
        self,
        messages: list[dict[str, str]],
        output_schema: type,
    ) -> Any:
        """Invoke the routed LLM with structured output.

        Returns a Pydantic model instance matching *output_schema*.
        """
        llm = self._router.get_structured_llm(self.AGENT_NAME, output_schema)
        return llm.invoke(messages)
