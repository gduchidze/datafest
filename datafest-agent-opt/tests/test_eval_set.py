"""Pins the 60-page eval set: the original 30 stay put, the new 30 are sound."""

from __future__ import annotations

import hashlib
import itertools
import re
from collections import Counter
from difflib import SequenceMatcher
from itertools import zip_longest

from doc_review_opt.agent.tools import check_prior_production
from doc_review_opt.experiment.live import LIVE_PAGE_LIMIT, pick_live_pages
from doc_review_opt.fixtures import CALIBRATION, EVAL_SET, HOLDOUT, PRIOR_PRODUCTION

ORIGINAL_IDS = (
    "ARC-000141",
    "ARC-000163",
    "ARC-000212",
    "ARC-000245",
    "ARC-000259",
    "ARC-000284",
    "ARC-000289",
    "ARC-000299",
    "ARC-000306",
    "ARC-000318",
    "ARC-000329",
    "ARC-000333",
    "ARC-000336",
    "ARC-000352",
    "ARC-000377",
    "ARC-000390",
    "ARC-000401",
    "ARC-000413",
    "ARC-000427",
    "ARC-000441",
    "ARC-000455",
    "ARC-000463",
    "ARC-000467",
    "ARC-000472",
    "ARC-000497",
    "ARC-000508",
    "ARC-000518",
    "ARC-000520",
    "ARC-000572",
    "ARC-000580",
)
ORIGINAL_CALIBRATION = (
    "ARC-000141",
    "ARC-000212",
    "ARC-000259",
    "ARC-000284",
    "ARC-000289",
    "ARC-000329",
    "ARC-000333",
    "ARC-000352",
    "ARC-000390",
    "ARC-000401",
    "ARC-000413",
    "ARC-000455",
    "ARC-000463",
    "ARC-000472",
    "ARC-000508",
    "ARC-000520",
)
ORIGINAL_HOLDOUT = (
    "ARC-000163",
    "ARC-000245",
    "ARC-000299",
    "ARC-000306",
    "ARC-000318",
    "ARC-000336",
    "ARC-000377",
    "ARC-000427",
    "ARC-000441",
    "ARC-000467",
    "ARC-000497",
    "ARC-000518",
    "ARC-000572",
    "ARC-000580",
)
MIN_LONG_CHARS = 3000
MAX_LONG_CHARS = 5000
MAX_SIMILARITY = 0.5
MAX_ORIGINAL_SIMILARITY = 0.6
FIELD_LABELS = ("employee id", "ssn", "phone", "email", "account", "dob")
ORIGINAL_DIGEST = "85e84f07d2e58c05bfae3e31526ae973f1a07ba44551aecc28fbad693be92b6b"
LIVE_IDS = (
    "ARC-000141",
    "ARC-000212",
    "ARC-000284",
    "ARC-000352",
    "ARC-000163",
    "ARC-000245",
)
ORIGINAL_ROUTES = {"produce": 9, "withhold": 6, "redact": 5, "attorney_review": 10}
ROUTE_ORDER = ("produce", "withhold", "redact", "attorney_review")
MIN_HARD_CASES = 12
NEW = EVAL_SET[len(ORIGINAL_IDS) :]
PHONE = re.compile(r"\b\d{3}-\d{4}\b")
EMAIL = re.compile(r"[\w.]+@[\w.-]+")


def test_eval_set_has_sixty_unique_pages() -> None:
    ids = [p.doc_id for p in EVAL_SET]
    assert len(ids) == len(set(ids)) == 60
    assert ids == sorted(ids)


def test_original_thirty_pages_are_unchanged_and_first() -> None:
    first = EVAL_SET[: len(ORIGINAL_IDS)]
    assert tuple(p.doc_id for p in first) == ORIGINAL_IDS
    digest = hashlib.sha256("".join(p.model_dump_json() for p in first).encode())
    assert digest.hexdigest() == ORIGINAL_DIGEST


def test_live_rerun_still_picks_the_same_pages() -> None:
    by_route: dict[str, list[str]] = {}
    for page in EVAL_SET:
        by_route.setdefault(page.labels.expected_route, []).append(page.doc_id)
    picked = [i for group in zip_longest(*by_route.values()) for i in group if i]
    assert tuple(picked[:LIVE_PAGE_LIMIT]) == LIVE_IDS
    assert tuple(p.doc_id for p in pick_live_pages(EVAL_SET)) == LIVE_IDS
    assert tuple(p.doc_id for p in pick_live_pages(EVAL_SET[:30])) == LIVE_IDS


def test_live_pages_are_one_per_route_in_turn_not_the_first_six() -> None:
    picked = pick_live_pages(EVAL_SET)
    assert {p.labels.expected_route for p in picked[:4]} == set(ROUTE_ORDER)
    assert {p.doc_id for p in picked} < set(ORIGINAL_IDS)
    assert tuple(p.doc_id for p in picked) != ORIGINAL_IDS[:LIVE_PAGE_LIMIT]


def test_new_pages_keep_the_route_mix_of_the_original_set() -> None:
    counts = Counter(p.labels.expected_route for p in NEW)
    assert len(NEW) == 30
    assert dict(counts) == ORIGINAL_ROUTES
    assert [r for r in ROUTE_ORDER if r in counts] == list(ROUTE_ORDER)


def test_new_set_adds_enough_hard_cases() -> None:
    hard = [p for p in NEW if p.labels.difficulty in ("trap", "borderline")]
    assert len(hard) >= MIN_HARD_CASES


def test_new_pages_cover_the_requested_hard_kinds() -> None:
    labels = [p.labels for p in NEW]
    cc_only = [p for p in NEW if p.cc and p.labels.expected_route == "produce"]
    assert cc_only
    assert any(
        not x.privileged and x.pii_spans and x.responsive_level == 0 for x in labels
    )
    assert any(
        x.responsive_level == 1 and x.expected_route == "attorney_review"
        for x in labels
    )
    assert any(x.privileged and x.difficulty in ("trap", "borderline") for x in labels)
    assert any(len(p.text) > 3000 and p.labels.pii_spans for p in NEW)
    assert any("check_prior_production" in x.expected_tools for x in labels)


def test_every_page_has_a_coherent_label_set() -> None:
    for page in EVAL_SET:
        labels = page.labels
        assert labels.contains_pii == bool(labels.pii_spans)
        assert all(span in page.text for span in labels.pii_spans)
        assert labels.must_escalate == (labels.expected_route == "attorney_review")
        if labels.privileged:
            assert labels.expected_route == "attorney_review"
        if labels.expected_route == "redact":
            assert labels.pii_spans
            assert labels.responsive_level == 2
        if labels.responsive_level == 0:
            assert labels.expected_route == "withhold"


def test_new_pages_are_obviously_fictional() -> None:
    for page in NEW:
        assert "ARCADIA FREIGHT" in page.text
        for phone in PHONE.findall(page.text):
            assert phone.startswith("555-")
        for address in EMAIL.findall(page.text):
            assert address.rstrip(".").endswith(".example")
        for span in page.labels.pii_spans:
            assert not re.fullmatch(r"(?!000|666|9)\d{3}-\d{2}-\d{4}", span)


def test_expected_prior_production_tools_resolve_to_an_index_entry() -> None:
    for page in EVAL_SET:
        if "check_prior_production" in page.labels.expected_tools:
            assert check_prior_production(page.doc_id)["found"]
    assert set(PRIOR_PRODUCTION) <= {p.doc_id for p in EVAL_SET}


def test_split_keeps_every_original_page_in_its_pinned_half() -> None:
    ids = {p.doc_id for p in (*CALIBRATION, *HOLDOUT)}
    assert ids == {p.doc_id for p in EVAL_SET}
    assert not {p.doc_id for p in CALIBRATION} & {p.doc_id for p in HOLDOUT}
    original = set(ORIGINAL_IDS)
    calibration = [p.doc_id for p in CALIBRATION if p.doc_id in original]
    holdout = [p.doc_id for p in HOLDOUT if p.doc_id in original]
    assert tuple(calibration) == ORIGINAL_CALIBRATION
    assert tuple(holdout) == ORIGINAL_HOLDOUT


def test_long_pages_are_not_near_duplicates_of_each_other() -> None:
    by_id = {p.doc_id: p for p in EVAL_SET}
    long_ids = [p.doc_id for p in NEW if len(p.text) > MIN_LONG_CHARS]
    assert len(long_ids) >= 2
    for first, second in itertools.combinations(long_ids, 2):
        ratio = SequenceMatcher(None, by_id[first].text, by_id[second].text).ratio()
        assert ratio < MAX_SIMILARITY
    assert all(len(by_id[i].text) <= MAX_LONG_CHARS for i in long_ids)


def test_pii_spans_are_personal_data_not_field_labels() -> None:
    for page in NEW:
        for span in page.labels.pii_spans:
            assert span == span.strip()
            assert not span.lower().startswith(FIELD_LABELS)


def test_level_one_escalations_are_damaged_pages() -> None:
    marks = ("illegible", "smudged", "cut off", "torn", "OCR")
    for page in NEW:
        labels = page.labels
        if labels.responsive_level == 1 and labels.expected_route == "attorney_review":
            assert any(mark in page.text for mark in marks)


def test_only_the_intended_duplicate_resembles_an_original_page() -> None:
    originals = EVAL_SET[: len(ORIGINAL_IDS)]
    for page in NEW:
        best = max(SequenceMatcher(None, page.text, o.text).ratio() for o in originals)
        if page.doc_id == "ARC-000649":
            assert best > MAX_ORIGINAL_SIMILARITY
        else:
            assert best < MAX_ORIGINAL_SIMILARITY
