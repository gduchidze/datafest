"""The routing policy that turns typed classifier answers into a route."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple

from doc_review_opt.config import get_settings
from doc_review_opt.models import (
    ClassifierResult,
    PredicateAnswer,
    RefusalAnswer,
    Route,
    ScoreAnswer,
)

SETTINGS = get_settings()


RouteReason = Literal[
    "privilege_above_threshold",
    "not_responsive",
    "ambiguous_responsiveness",
    "low_responsive_confidence",
    "pii_above_threshold",
    "clean_responsive",
    "agent_disposition",
    "classifier_error",
    "missing_or_refused_answer",
    "redaction_failed",
]


@dataclass(frozen=True)
class RoutingPolicy:
    privilege_escalation: float = SETTINGS.policy.default.privilege_escalation
    pii_redaction: float = SETTINGS.policy.default.pii_redaction
    not_responsive: float = SETTINGS.policy.default.not_responsive
    responsive: float = SETTINGS.policy.default.responsive
    responsive_confidence: float = SETTINGS.policy.default.responsive_confidence


DEFAULT_POLICY = RoutingPolicy()


class RouteDecision(NamedTuple):
    route: Route
    reason: RouteReason


def _typed_answers(
    result: ClassifierResult,
) -> tuple[ScoreAnswer, PredicateAnswer, PredicateAnswer] | None:
    responsive = result.answers.get("responsive")
    privileged = result.answers.get("privileged")
    pii = result.answers.get("contains_pii")
    if (
        isinstance(responsive, ScoreAnswer)
        and isinstance(privileged, PredicateAnswer)
        and isinstance(pii, PredicateAnswer)
    ):
        return responsive, privileged, pii
    return None


def _route_agent(
    disposition: Route, result: ClassifierResult, policy: RoutingPolicy
) -> RouteDecision:
    privileged = result.answers.get("privileged")
    if (
        isinstance(privileged, PredicateAnswer)
        and privileged.probability > policy.privilege_escalation
    ):
        return RouteDecision("attorney_review", "privilege_above_threshold")
    return RouteDecision(disposition, "agent_disposition")


def decide(
    result: ClassifierResult, policy: RoutingPolicy = DEFAULT_POLICY
) -> RouteDecision:
    if result.error is not None:
        return RouteDecision("attorney_review", "classifier_error")
    if any(isinstance(a, RefusalAnswer) for a in result.answers.values()):
        return RouteDecision("attorney_review", "missing_or_refused_answer")
    if result.agent_disposition is not None:
        return _route_agent(result.agent_disposition, result, policy)
    typed = _typed_answers(result)
    if typed is None:
        return RouteDecision("attorney_review", "missing_or_refused_answer")
    responsive, privileged, pii = typed
    if privileged.probability > policy.privilege_escalation:
        return RouteDecision("attorney_review", "privilege_above_threshold")
    if responsive.score < policy.not_responsive:
        return RouteDecision("withhold", "not_responsive")
    if responsive.score < policy.responsive:
        return RouteDecision("attorney_review", "ambiguous_responsiveness")
    if responsive.confidence < policy.responsive_confidence:
        return RouteDecision("attorney_review", "low_responsive_confidence")
    if pii.probability > policy.pii_redaction:
        return RouteDecision("redact", "pii_above_threshold")
    return RouteDecision("produce", "clean_responsive")


def flags_pii(result: ClassifierResult, policy: RoutingPolicy = DEFAULT_POLICY) -> bool:
    pii = result.answers.get("contains_pii")
    if isinstance(pii, PredicateAnswer):
        return (
            pii.probability > policy.pii_redaction
            or result.agent_disposition == "redact"
            or result.agent_redact is True
        )
    return result.agent_redact is not False


def route(result: ClassifierResult, policy: RoutingPolicy = DEFAULT_POLICY) -> Route:
    return decide(result, policy).route


__all__ = [
    "DEFAULT_POLICY",
    "RouteDecision",
    "RoutingPolicy",
    "decide",
    "flags_pii",
    "route",
]
