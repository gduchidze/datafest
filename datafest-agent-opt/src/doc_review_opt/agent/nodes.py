"""Graph nodes: classify, the terminal steps and attorney review."""

from __future__ import annotations

import logging
from typing import (
    Any,
    Literal,
    NamedTuple,
)

from langgraph.types import Command, interrupt

from doc_review_opt.agent.protocols import Classifier, Redactor
from doc_review_opt.agent.redactor import (
    RedactionError,
)
from doc_review_opt.agent.routing import (
    RoutingPolicy,
    decide,
    flags_pii,
)
from doc_review_opt.agent.state import (
    PageState,
    interrupt_payload,
)
from doc_review_opt.models import (
    BilledSpend,
    PageResult,
    RedactionResult,
)

logger = logging.getLogger(__name__)


ATTORNEY_DECISIONS = ("produce", "withhold")


class InvalidAttorneyDecisionError(ValueError): ...


class _AttorneyText(NamedTuple):
    text: str | None
    redaction: RedactionResult | None
    decision: Literal["produce", "withhold"]
    failed_redaction: BilledSpend


def _spent_by(error: RedactionError) -> BilledSpend:
    return BilledSpend(usage=error.usage, cost_usd=error.cost_usd)


class _PageNodes:
    def __init__(
        self, classifier: Classifier, redactor: Redactor, policy: RoutingPolicy
    ) -> None:
        self._classifier = classifier
        self._redactor = redactor
        self._policy = policy

    async def classify(self, state: PageState) -> dict[str, Any]:
        result = await self._classifier.classify(state["page"], state["protocol"])
        decision = decide(result, self._policy)
        return {
            "classification": result,
            "route": decision.route,
            "reason": decision.reason,
        }

    async def attorney_review(self, state: PageState) -> dict[str, Any]:
        payload = interrupt_payload(
            state["doc_id"], state["reason"], state["classification"]
        )
        decision = interrupt(payload)
        if decision not in ATTORNEY_DECISIONS:
            raise InvalidAttorneyDecisionError(
                f"{state['doc_id']}: {decision!r} is not one of {ATTORNEY_DECISIONS}"
            )
        produced = await self._attorney_text(state, decision)
        result = PageResult(
            doc_id=state["doc_id"],
            decision=produced.decision,
            route="attorney_review",
            redacted=produced.redaction is not None,
            produced_text=produced.text,
            escalated=True,
            attorney_decision=decision,
            classification=state["classification"],
            redaction=produced.redaction,
            failed_redaction=produced.failed_redaction,
            reason=state["reason"],
        )
        return {"results": [result]}

    async def _attorney_text(self, state: PageState, decision: str) -> _AttorneyText:
        earlier = state.get("failed_redaction", BilledSpend())
        if decision != "produce":
            return _AttorneyText(None, None, "withhold", earlier)
        page = state["page"]
        if not flags_pii(state["classification"], self._policy):
            return _AttorneyText(page.text, None, "produce", earlier)
        try:
            redaction = await self._redactor.redact(page)
        except RedactionError as error:
            logger.warning(
                "redaction failed doc_id=%s error=%s", state["doc_id"], error
            )
            return _AttorneyText(None, None, "withhold", earlier + _spent_by(error))
        return _AttorneyText(redaction.text, redaction, "produce", earlier)

    async def redact(
        self, state: PageState
    ) -> Command[Literal["attorney_review"]] | dict[str, Any]:
        try:
            redaction = await self._redactor.redact(state["page"])
        except RedactionError as error:
            logger.warning(
                "redaction failed doc_id=%s error=%s", state["doc_id"], error
            )
            return Command(
                goto="attorney_review",
                update={
                    "route": "attorney_review",
                    "reason": "redaction_failed",
                    "failed_redaction": _spent_by(error),
                },
            )
        result = PageResult(
            doc_id=state["doc_id"],
            decision="produce",
            route="redact",
            redacted=True,
            produced_text=redaction.text,
            classification=state["classification"],
            redaction=redaction,
            reason=state["reason"],
        )
        return {"results": [result]}

    async def produce(self, state: PageState) -> dict[str, Any]:
        result = PageResult(
            doc_id=state["doc_id"],
            decision="produce",
            route="produce",
            produced_text=state["page"].text,
            classification=state["classification"],
            reason=state["reason"],
        )
        return {"results": [result]}

    async def withhold(self, state: PageState) -> dict[str, Any]:
        result = PageResult(
            doc_id=state["doc_id"],
            decision="withhold",
            route="withhold",
            classification=state["classification"],
            reason=state["reason"],
        )
        return {"results": [result]}
