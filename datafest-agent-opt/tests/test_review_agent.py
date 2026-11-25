"""Tests for the synthetic eval set and mock tools."""

import pytest

from doc_review_opt.agent.prompts import STABLE_REVIEW_PREFIX
from doc_review_opt.agent.tools import (
    SLIM_TOOLS,
    VERBOSE_TOOLS,
    UnknownToolError,
    allowed_tool_choice,
    execute_local_tool,
)
from doc_review_opt.fixtures import ATTORNEYS, CALIBRATION, EVAL_SET, HOLDOUT, PROTOCOL
from doc_review_opt.models import PageLabels, ReviewPage
from doc_review_opt.pricing import approx_tokens

LABELS = [(page.doc_id, page.labels) for page in EVAL_SET]


def test_doc_ids_are_unique() -> None:
    ids = [page.doc_id for page in EVAL_SET]
    assert len(ids) == len(set(ids)) == 60


def test_splits_are_disjoint_and_cover_the_set() -> None:
    calibration = {p.doc_id for p in CALIBRATION}
    holdout = {p.doc_id for p in HOLDOUT}
    assert not calibration & holdout
    assert calibration | holdout == {p.doc_id for p in EVAL_SET}


def test_every_multi_page_route_is_in_both_splits() -> None:
    routes = {p.labels.expected_route for p in EVAL_SET}
    assert {p.labels.expected_route for p in CALIBRATION} == routes
    assert {p.labels.expected_route for p in HOLDOUT} == routes


@pytest.mark.parametrize("page", EVAL_SET, ids=lambda p: p.doc_id)
def test_pii_spans_appear_in_the_page_text(page: ReviewPage) -> None:
    assert all(span in page.text for span in page.labels.pii_spans)
    assert page.labels.contains_pii == bool(page.labels.pii_spans)


@pytest.mark.parametrize(("doc_id", "labels"), LABELS)
def test_labels_are_consistent(doc_id: str, labels: PageLabels) -> None:
    route = labels.expected_route
    assert labels.expected_disposition == ("produce" if route == "redact" else route)
    assert labels.must_escalate == (route == "attorney_review")
    if labels.privileged:
        assert route == "attorney_review"
    if route == "redact":
        assert labels.contains_pii
        assert labels.responsive_level == 2
    if labels.responsive_level == 0:
        assert route == "withhold"


def test_the_gist_pages_keep_their_text_and_routes() -> None:
    by_id = {p.doc_id: p for p in EVAL_SET}
    routes = {
        "ARC-000141": "produce",
        "ARC-000212": "withhold",
        "ARC-000377": "redact",
        "ARC-000401": "attorney_review",
        "ARC-000455": "attorney_review",
        "ARC-000508": "produce",
    }
    assert {k: by_id[k].labels.expected_route for k in routes} == routes
    assert "payroll account 8841-220-19" in by_id["ARC-000377"].text


def test_attorney_names_match_the_protocol() -> None:
    assert all(name in PROTOCOL for name in ATTORNEYS)


def test_protocol_is_long_enough_to_cache() -> None:
    assert approx_tokens(PROTOCOL) >= 1100
    assert STABLE_REVIEW_PREFIX.endswith(PROTOCOL)


def test_slim_tools_are_smaller_than_verbose_tools() -> None:
    assert approx_tokens(SLIM_TOOLS) < approx_tokens(VERBOSE_TOOLS)
    assert [t["name"] for t in SLIM_TOOLS] == [t["name"] for t in VERBOSE_TOOLS]


def test_slim_payload_is_smaller_than_verbose_payload() -> None:
    args = {"name": "Priya Raghunathan"}
    slim = execute_local_tool("lookup_custodian", args, "slim")
    verbose = execute_local_tool("lookup_custodian", args, "verbose")
    assert slim == {"found": True, "name": "Priya Raghunathan", "is_attorney": True}
    assert approx_tokens(verbose) > 10 * approx_tokens(slim)


def test_prior_production_finds_the_duplicate() -> None:
    result = execute_local_tool("check_prior_production", {"doc_id": "ARC-000572"})
    assert result["matched_doc_id"] == "ARC-000141"


def test_escalation_note_carries_no_page_text() -> None:
    result = execute_local_tool(
        "escalate_to_attorney", {"doc_id": "ARC-000401", "reason": "attorney author"}
    )
    assert result["queue"] == "attorney_review"


def test_unknown_tool_is_rejected() -> None:
    with pytest.raises(UnknownToolError):
        execute_local_tool("delete_everything", {})


def test_allowed_tool_choice_shape() -> None:
    choice = allowed_tool_choice(["lookup_custodian"])
    assert choice["tools"] == [{"type": "function", "name": "lookup_custodian"}]
    assert choice["mode"] == "auto"
