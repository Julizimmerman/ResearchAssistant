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

from research_assistant.models import CostRecord
from research_assistant.routing import ModelRouter

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class providing shared infrastructure for every agent.

    Subclasses must set ``AGENT_NAME`` and implement ``run()``.
    """

    AGENT_NAME: str = ""  # overridden by each subclass

    def __init__(self, router: ModelRouter) -> None:
        self._router = router

    # ── Abstract interface ───────────────────────────────────────────

    @abstractmethod
    def run(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the agent's task and return a partial state dict."""
        ...

    # ── LLM invocation helpers ───────────────────────────────────────

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
        task_description: str = "",
    ) -> tuple[Any, CostRecord]:
        """Invoke the routed LLM with structured output and record cost.

        Returns a ``(parsed_object, cost_record)`` tuple.  The parsed object
        is a Pydantic model instance matching *output_schema*.
        """
        llm = self._router.get_structured_llm(self.AGENT_NAME, output_schema)
        result = llm.invoke(messages)

        # ``with_structured_output(include_raw=True)`` returns
        # ``{"raw": AIMessage, "parsed": PydanticModel}``.
        # In mock mode the mock class returns the same shape.
        if isinstance(result, dict) and "parsed" in result:
            parsed = result["parsed"]
            raw = result["raw"]
            usage = getattr(raw, "usage_metadata", None) or {}
        else:
            # Fallback: the result itself is the Pydantic object (mock shortcut).
            parsed = result
            usage = {}

        prompt_tokens = usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)
        total_tokens = prompt_tokens + completion_tokens

        model_name = self._router.get_model_name(self.AGENT_NAME)
        cost_record = CostRecord(
            agent_name=self.AGENT_NAME,
            model_name=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=self._router.cost_tracker.calculate_cost(
                model_name, prompt_tokens, completion_tokens
            ),
            task_description=task_description,
        )
        self._router.cost_tracker.add_record(cost_record)

        return parsed, cost_record
