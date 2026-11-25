"""Tests for the plain-code router."""

import pytest

from doc_review_opt.agent.routing import RoutingPolicy, decide, flags_pii, route
from doc_review_opt.models import ClassifierResult
from tests.helpers import make_result, refused


def test_clean_responsive_page_is_produced() -> None:
    result = make_result(responsive=2.0, confidence=0.95, privileged=0.01, pii=0.02)
    assert route(result) == "produce"


def test_non_responsive_page_is_withheld() -> None:
    result = make_result(responsive=0.1, confidence=0.95, privileged=0.01, pii=0.02)
    assert route(result) == "withhold"


def test_fractional_score_just_under_threshold_is_withheld() -> None:
    result = make_result(responsive=0.49, confidence=0.95, privileged=0.01, pii=0.02)
    assert route(result) == "withhold"


def test_pii_page_is_routed_to_redaction() -> None:
    result = make_result(responsive=2.0, confidence=0.95, privileged=0.01, pii=0.91)
    assert route(result) == "redact"


@pytest.mark.parametrize(
    ("privileged", "confidence"),
    [(0.25, 0.95), (0.01, 0.5)],
    ids=["privilege_above_threshold", "low_responsive_confidence"],
)
def test_uncertain_pages_escalate(privileged: float, confidence: float) -> None:
    result = make_result(
        responsive=2.0, confidence=confidence, privileged=privileged, pii=0.0
    )
    assert route(result) == "attorney_review"


def test_privilege_beats_pii() -> None:
    result = make_result(responsive=2.0, confidence=0.99, privileged=0.9, pii=0.99)
    assert route(result) == "attorney_review"


def test_ambiguous_band_escalates_despite_high_confidence() -> None:
    result = make_result(responsive=1.2, confidence=0.98, privileged=0.01, pii=0.0)
    assert decide(result).reason == "ambiguous_responsiveness"


def test_decisive_score_survives_mediocre_confidence() -> None:
    result = make_result(responsive=0.07, confidence=0.55, privileged=0.01, pii=0.0)
    assert route(result) == "withhold"


def test_custom_policy_changes_the_boundary() -> None:
    result = make_result(responsive=2.0, confidence=0.95, privileged=0.1, pii=0.0)
    assert route(result) == "produce"
    assert route(result, RoutingPolicy(privilege_escalation=0.05)) == "attorney_review"


@pytest.mark.parametrize("name", ["responsive", "privileged", "contains_pii"])
def test_refusal_fails_safe_to_attorney_review(name: str) -> None:
    base = make_result(responsive=2.0, confidence=0.99, privileged=0.0, pii=0.0)
    result = base.model_copy(update={"answers": {**base.answers, name: refused(name)}})
    assert decide(result).reason == "missing_or_refused_answer"


def test_missing_answer_fails_safe() -> None:
    result = ClassifierResult(engine="x", model="y")
    assert route(result) == "attorney_review"


def test_error_fails_safe() -> None:
    result = make_result(responsive=2.0, confidence=0.99, privileged=0.0, pii=0.0)
    failed = result.model_copy(update={"error": "boom"})
    assert decide(failed).reason == "classifier_error"


def test_agent_disposition_is_honoured() -> None:
    result = ClassifierResult(engine="agent", model="m", agent_disposition="withhold")
    assert route(result) == "withhold"


def test_privilege_safety_still_applies_to_an_agent_that_says_produce() -> None:
    typed = make_result(responsive=2.0, confidence=0.9, privileged=0.8, pii=0.0)
    result = typed.model_copy(update={"agent_disposition": "produce"})
    assert route(result) == "attorney_review"


def test_agent_can_route_to_redaction() -> None:
    result = ClassifierResult(
        engine="agent", model="m", agent_disposition="redact", answers={}
    )
    assert decide(result).route == "redact"


def test_agent_redact_still_yields_to_typed_privilege() -> None:
    base = make_result(responsive=2.0, confidence=0.9, privileged=0.9, pii=0.0)
    result = base.model_copy(update={"agent_disposition": "redact"})
    assert route(result) == "attorney_review"


def test_flags_pii_fails_closed_without_a_typed_signal() -> None:
    typed = make_result(responsive=2.0, confidence=0.9, privileged=0.0, pii=0.5)
    assert not flags_pii(typed)
    assert flags_pii(
        make_result(responsive=2.0, confidence=0.9, privileged=0.0, pii=0.6)
    )
    assert flags_pii(ClassifierResult(engine="x", model="m", error="boom"))
    unparsed = ClassifierResult(
        engine="agent", model="m", agent_disposition="attorney_review"
    )
    assert flags_pii(unparsed)
    assert flags_pii(unparsed.model_copy(update={"agent_redact": True}))
    assert not flags_pii(unparsed.model_copy(update={"agent_redact": False}))
