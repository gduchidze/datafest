"""Package-wide exceptions."""

from __future__ import annotations

from doc_review_opt.models import TokenUsage


class UnusableReplyError(ValueError): ...


class BudgetExceededError(RuntimeError): ...


class ConfigError(ValueError): ...


class DiagramLayoutError(ValueError): ...


class BilledCallError(RuntimeError):
    def __init__(
        self, message: str, usage: TokenUsage | None = None, cost_usd: float = 0.0
    ) -> None:
        super().__init__(message)
        self.usage = usage or TokenUsage()
        self.cost_usd = cost_usd
