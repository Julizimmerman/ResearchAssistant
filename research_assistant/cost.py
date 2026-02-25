"""Cost tracking for LLM usage across all agents.

Maintains a running ledger of every LLM call — model used, tokens consumed,
and estimated USD cost — so the system can display a summary at the end of
each research session.
"""

from __future__ import annotations

from research_assistant.models import CostRecord

# Pricing per 1 million tokens (USD).  Update when OpenAI changes prices.
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
    "gpt-4o": {"prompt": 2.50, "completion": 10.00},
}


class CostTracker:
    """Accumulates ``CostRecord`` entries and provides aggregate summaries."""

    def __init__(self) -> None:
        self._records: list[CostRecord] = []

    # ── Recording ────────────────────────────────────────────────────

    def calculate_cost(
        self,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        """Return estimated USD cost for a single LLM call."""
        pricing = MODEL_PRICING.get(model_name, {"prompt": 0.0, "completion": 0.0})
        prompt_cost = (prompt_tokens / 1_000_000) * pricing["prompt"]
        completion_cost = (completion_tokens / 1_000_000) * pricing["completion"]
        return prompt_cost + completion_cost

    def add_record(self, record: CostRecord) -> None:
        """Append a cost record to the ledger."""
        self._records.append(record)

    # ── Queries ──────────────────────────────────────────────────────

    @property
    def records(self) -> list[CostRecord]:
        """Return a copy of all recorded entries."""
        return list(self._records)

    @property
    def total_cost(self) -> float:
        """Total estimated USD across every recorded call."""
        return sum(r.estimated_cost_usd for r in self._records)

    def summary_by_agent(self) -> dict[str, float]:
        """Return total estimated cost grouped by agent name."""
        result: dict[str, float] = {}
        for r in self._records:
            result[r.agent_name] = result.get(r.agent_name, 0.0) + r.estimated_cost_usd
        return result

    def summary_by_model(self) -> dict[str, dict[str, int | float]]:
        """Return call count, total tokens, and cost grouped by model name."""
        result: dict[str, dict[str, int | float]] = {}
        for r in self._records:
            if r.model_name not in result:
                result[r.model_name] = {"calls": 0, "tokens": 0, "cost": 0.0}
            result[r.model_name]["calls"] += 1
            result[r.model_name]["tokens"] += r.total_tokens
            result[r.model_name]["cost"] += r.estimated_cost_usd
        return result
