"""Synthetic matter, custodians, protocol, prior production and labeled eval pages."""

from __future__ import annotations

import textwrap
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from doc_review_opt.models import (
    Difficulty,
    PageLabels,
    ReviewPage,
    Risk,
    Route,
)

MATTER = "24-CV-118 (Hollis v. Arcadia Freight)"


@dataclass(frozen=True)
class Custodian:
    name: str
    title: str
    department: str
    is_attorney: bool


CUSTODIANS: Mapping[str, Custodian] = {
    c.name: c
    for c in (
        Custodian("Dana Whitfield", "Controller", "Finance", False),
        Custodian("Marcus Oyelaran", "VP Finance", "Finance", False),
        Custodian("Yusuf Demir", "Revenue Accountant", "Finance", False),
        Custodian("Raj Venkatesan", "Treasury Manager", "Finance", False),
        Custodian("Nadia Brooks", "Chief Financial Officer", "Executive", False),
        Custodian("Lena Ortiz", "Corporate Secretary", "Executive", False),
        Custodian("Owen Fitch", "Payroll Manager", "Human Resources", False),
        Custodian("Grace Lindqvist", "HR Business Partner", "Human Resources", False),
        Custodian("Ines Calloway", "Director of Facilities", "Operations", False),
        Custodian("Priya Raghunathan", "Associate General Counsel", "Legal", True),
        Custodian("Tomas Brandt", "General Counsel", "Legal", True),
        Custodian(
            "Helen Park", "Outside Counsel, Kessler & Moore LLP", "External", True
        ),
    )
}


ATTORNEYS = tuple(c.name for c in CUSTODIANS.values() if c.is_attorney)


PROTOCOL_SECTIONS: Mapping[str, str] = {
    "responsiveness": textwrap.dedent(
        """
        Matter 24-CV-118 (Hollis v. Arcadia Freight). Produce all documents concerning
        Q3 2024 revenue recognition, including revenue timing, cutoff adjustments, and
        communications about restating quarterly figures.

        A page is responsive (level 2) when it directly concerns revenue recognition
        for the quarter ended 30 September 2024: recognized revenue totals, the
        timing of shipments against the period end, the $3.1M cutoff adjustment,
        supporting workpapers and journal entries, committee or board discussion of
        the cutoff convention, and any discussion of restating quarterly figures.
        A page is possibly responsive (level 1) when it touches the subject only
        indirectly, or when it is too damaged or too short to tell. A page is not
        responsive (level 0) when it is unrelated to the matter: facilities notices,
        social events, IT notices, benefits administration and similar housekeeping.
        Mentioning a person who appears in the matter does not make a page
        responsive. Judge the page by its subject, not by who wrote it.
        """
    ).strip(),
    "privilege": textwrap.dedent(
        """
        Attorney-client privilege and work product. The attorneys on this matter are
        Priya Raghunathan (Associate General Counsel), Tomas Brandt (General
        Counsel) and Helen Park (outside counsel, Kessler & Moore LLP). Everyone else
        in the custodian directory is a non-attorney, including the Controller, the
        CFO and the Corporate Secretary.

        A page is privileged when a named attorney is its author or a direct
        recipient (From or To), or when its text relays legal advice or litigation
        strategy, even second hand and even without a privilege header. A page is
        not privileged merely because an attorney is copied (cc) on routine business
        correspondence that asks for no legal advice. A page is not privileged
        because it discusses a restatement, an audit, or regulatory exposure in
        business terms with no attorney involved. A privilege header with no
        attorney and no advice does not make a page privileged.

        Never produce a privileged page. Producing one is far costlier than an
        unnecessary attorney review, so when privilege is plausible the page goes to
        an attorney. A forwarded copy of a privileged page is treated as privileged.
        """
    ).strip(),
    "pii": textwrap.dedent(
        """
        Personal information must not leave the building. Personal information means
        personal email addresses and direct phone numbers, home addresses, dates of
        birth, government identifiers such as Social Security numbers, employee IDs,
        bank and payroll account numbers, driver license numbers, and health or
        medical information about an individual. Business titles, company names,
        and invoice or workpaper reference numbers are not personal information.

        A responsive page that contains personal information is produced only after
        every piece of it has been replaced by the literal token [REDACTED]. Redact
        the underlying characters; do not mask, abbreviate or summarize them, and
        keep all other text exactly as written. A page that is not responsive is
        withheld whole and needs no redaction, even if it is full of personal data.
        """
    ).strip(),
    "production_format": textwrap.dedent(
        """
        Production format and duplicates. Each page is reviewed on its own and ends
        in exactly one disposition: produce, withhold, or attorney review. An exact
        or near duplicate of a page that an earlier production volume already
        contains is withheld as a duplicate, so the same page is never produced
        twice; check the prior production index for the control number. Pages sent
        to attorney review are decided by the attorney as produce or withhold.
        Pages too damaged to read, such as partial OCR of handwriting, go to
        attorney review rather than being guessed at. Escalation notes carry the
        control number and the reason only; they never quote the page, because the
        note itself is not privileged.
        """
    ).strip(),
    "review_guidelines": textwrap.dedent(
        """
        Reviewer guidelines. Decide from the page and its header lines (author,
        recipients, cc) alone; do not speculate about documents you have not seen.
        When a page mixes a responsive discussion with a housekeeping remark, judge
        the whole page by its main subject. When two readings are both plausible and
        one of them is privileged, take the privileged reading and escalate. When a
        page is responsive and clean, produce it as is; do not redact business
        titles, company names, dollar amounts or workpaper numbers, because
        over-redaction makes the production useless to the requesting party.
        Responsiveness is judged against the matter scope above, not against how
        sensitive or embarrassing a page is. A sensitive page that is unrelated to
        the matter is withheld; a dull page that directly concerns revenue
        recognition for the quarter is produced. Short pages, handwritten notes and
        scans are reviewed under the same rules, and anything that cannot be read
        well enough to apply them goes to attorney review. Record the reason for an
        escalation in one short sentence that names the rule and never repeats the
        page text. A reviewer who is unsure whether a name belongs to an attorney
        checks the custodian directory instead of guessing from the job title.
        """
    ).strip(),
}


PROTOCOL = "\n\n".join(
    f"{name.upper().replace('_', ' ')}\n{body}"
    for name, body in PROTOCOL_SECTIONS.items()
)


def _labels(
    route: Route,
    level: Literal[0, 1, 2],
    *,
    difficulty: Difficulty = "simple",
    risk: Risk = "low",
    privileged: bool = False,
    pii: tuple[str, ...] = (),
    tools: tuple[str, ...] = (),
) -> PageLabels:
    return PageLabels(
        expected_disposition="produce" if route == "redact" else route,
        expected_route=route,
        responsive_level=level,
        privileged=privileged,
        contains_pii=bool(pii),
        pii_spans=pii,
        must_escalate=route == "attorney_review",
        difficulty=difficulty,
        risk=risk,
        expected_tools=tools,
    )


def _page(
    doc_id: str,
    custodian: str,
    author: str,
    text: str,
    labels: PageLabels,
    *,
    to: tuple[str, ...] = (),
    cc: tuple[str, ...] = (),
) -> ReviewPage:
    return ReviewPage(
        doc_id=doc_id,
        custodian=custodian,
        author=author,
        recipients=to,
        cc=cc,
        text=text,
        labels=labels,
    )


_PRIV_TOOLS = ("lookup_custodian", "escalate_to_attorney")


_GIST_PAGES = (
    _page(
        "ARC-000141",
        "Dana Whitfield",
        "Finance Reporting",
        "ARCADIA FREIGHT -- Q3 2024 REVENUE SUMMARY (INTERNAL)\n"
        "Recognized revenue for the quarter ended 30 Sep 2024 was $48.2M, up 6% "
        "sequentially. Of this, $3.1M relates to shipments that departed on "
        "1 Oct and was recognized in Q3 under the prior cutoff convention. "
        "Finance is reviewing whether the cutoff should be restated.",
        _labels("produce", 2),
    ),
    _page(
        "ARC-000212",
        "Ines Calloway",
        "Ines Calloway",
        "ARCADIA FREIGHT -- FACILITIES NOTICE\n"
        "The Dock 4 loading bay will be closed for resurfacing from 11-14 March. "
        "Please route all inbound pallets to Dock 2. Vending machines on the "
        "second floor will be restocked Tuesday.",
        _labels("withhold", 0),
    ),
    _page(
        "ARC-000377",
        "Dana Whitfield",
        "Dana Whitfield",
        "ARCADIA FREIGHT -- Q3 CUTOFF ADJUSTMENT APPROVALS\n"
        "The Q3 revenue cutoff adjustment of $3.1M was approved by the following:\n"
        "  Dana Whitfield, Controller -- dana.whitfield@arcadiafreight.example "
        "-- mobile 555-0148 -- employee ID 44192\n"
        "  Marcus Oyelaran, VP Finance -- payroll account 8841-220-19\n"
        "Supporting workpapers are filed under WP-Q3-CUTOFF-11.",
        _labels(
            "redact",
            2,
            risk="medium",
            pii=(
                "dana.whitfield@arcadiafreight.example",
                "555-0148",
                "44192",
                "8841-220-19",
            ),
            tools=("apply_redactions",),
        ),
    ),
    _page(
        "ARC-000401",
        "Marcus Oyelaran",
        "Priya Raghunathan",
        "PRIVILEGED AND CONFIDENTIAL -- ATTORNEY-CLIENT COMMUNICATION\n"
        "From: Priya Raghunathan, Associate General Counsel\n"
        "To: Marcus Oyelaran, VP Finance\n"
        "Marcus -- my advice is that the Q3 cutoff treatment is defensible but "
        "not free of risk. If we restate, we should expect the plaintiffs to "
        "argue the original recognition was knowing. Do not circulate this note "
        "outside the finance leadership group.",
        _labels(
            "attorney_review",
            2,
            risk="high",
            privileged=True,
            tools=_PRIV_TOOLS,
        ),
        to=("Marcus Oyelaran",),
    ),
    _page(
        "ARC-000455",
        "Dana Whitfield",
        "Unknown",
        "ARCADIA FREIGHT -- HANDWRITTEN NOTE (OCR, PARTIAL)\n"
        "[illegible] ...cutoff?? ask P before we book it -- she said last time "
        "we [illegible] exposure. 3.1 or 3.4, check w/ Dana. don't put in deck",
        _labels(
            "attorney_review",
            1,
            difficulty="borderline",
            risk="medium",
            tools=("escalate_to_attorney",),
        ),
    ),
    _page(
        "ARC-000508",
        "Lena Ortiz",
        "Lena Ortiz",
        "ARCADIA FREIGHT -- AUDIT COMMITTEE MINUTES (EXCERPT)\n"
        "The Committee discussed the Q3 revenue cutoff convention and asked "
        "management to prepare a memo quantifying the effect of recognizing the "
        "$3.1M in Q4 instead. No decision was taken. Next review: 14 November.",
        _labels("produce", 2),
    ),
)


_NEW_PAGES = (
    _page(
        "ARC-000163",
        "Dana Whitfield",
        "Yusuf Demir",
        "ARCADIA FREIGHT -- EMAIL\n"
        "From: Yusuf Demir, Revenue Accountant\n"
        "To: Dana Whitfield, Controller\n"
        "Subject: Q3 cutoff journal entry\n"
        "Dana -- journal entry JE-2024-0931 reverses the $3.1M of 1 Oct shipments "
        "out of Q3 revenue and re-books it in Q4. Supporting schedule is "
        "WP-Q3-CUTOFF-11. Please approve before the close calendar locks on Friday.",
        _labels("produce", 2),
        to=("Dana Whitfield",),
    ),
    _page(
        "ARC-000289",
        "Yusuf Demir",
        "Yusuf Demir",
        "ARCADIA FREIGHT -- Q3 CLOSE CHECKLIST (EXCERPT)\n"
        "Item 14: Shipments in transit at period end -- confirm revenue cutoff test "
        "against bills of lading dated 30 Sep to 3 Oct. Status: OPEN. "
        "Item 15: Reconcile recognized revenue to the sub-ledger ($48.2M). "
        "Status: complete.",
        _labels("produce", 2),
    ),
    _page(
        "ARC-000336",
        "Nadia Brooks",
        "Nadia Brooks",
        "ARCADIA FREIGHT -- BOARD FINANCE DECK, SPEAKER NOTES\n"
        "Slide 12: Q3 revenue of $48.2M includes a $3.1M cutoff timing item. If "
        "the quarter is restated the item moves to Q4 and Q3 growth falls from 6% "
        "to roughly 0%. Management recommends waiting for the auditors' view.",
        _labels("produce", 2),
    ),
    _page(
        "ARC-000245",
        "Grace Lindqvist",
        "Grace Lindqvist",
        "ARCADIA FREIGHT -- HR ANNOUNCEMENT\n"
        "The annual holiday party is on 12 December at the Harborview Hall. "
        "Please RSVP by 1 December. Vegetarian and gluten-free options will be "
        "available. Carpool sign-up sheet is at reception.",
        _labels("withhold", 0),
    ),
    _page(
        "ARC-000318",
        "Ines Calloway",
        "IT Service Desk",
        "ARCADIA FREIGHT -- IT NOTICE\n"
        "Network maintenance window: Saturday 02:00-04:00. The guest Wi-Fi and "
        "badge readers on Dock 2 will be offline. Rotate your VPN token before "
        "Friday using the self-service portal.",
        _labels("withhold", 0),
    ),
    _page(
        "ARC-000352",
        "Yusuf Demir",
        "Noor Haddad",
        "ARCADIA FREIGHT -- SALES OPS EMAIL\n"
        "Customer shipments affected by the Q3 cutoff, $3.1M in total. The largest "
        "is the Northpoint Mills lot ($1.4M, departed 1 Oct). For delivery "
        "confirmation call the receiving contact, Wendell Okafor, on 555-0173 or "
        "at wendell.okafor@northpoint-mills.example before we finalize timing.",
        _labels(
            "redact",
            2,
            risk="medium",
            pii=("555-0173", "wendell.okafor@northpoint-mills.example"),
            tools=("apply_redactions",),
        ),
    ),
    _page(
        "ARC-000390",
        "Owen Fitch",
        "Owen Fitch",
        "ARCADIA FREIGHT -- PAYROLL ACCRUAL NOTE\n"
        "The Q3 incentive plan pays out only if recognized Q3 revenue is at least "
        "$48.0M; the $3.1M cutoff item decides that threshold. Accrual for the "
        "Controller's bonus: $12,500 (SSN 900-41-7782). Hold the accrual until the "
        "cutoff question is settled.",
        _labels(
            "redact",
            2,
            risk="medium",
            pii=("900-41-7782",),
            tools=("apply_redactions",),
        ),
    ),
    _page(
        "ARC-000427",
        "Raj Venkatesan",
        "Raj Venkatesan",
        "ARCADIA FREIGHT -- TREASURY MEMO\n"
        "If the $3.1M cutoff adjustment stands, a customer overpayment of $42,300 "
        "must be refunded. Wire to the customer's account 0042-7781-3309 at "
        "Lakeshore Credit Union, and note the refund against Q3 revenue "
        "recognition once the adjustment is booked.",
        _labels(
            "redact",
            2,
            risk="medium",
            pii=("0042-7781-3309",),
            tools=("apply_redactions",),
        ),
    ),
    _page(
        "ARC-000463",
        "Dana Whitfield",
        "Dana Whitfield",
        "ARCADIA FREIGHT -- OUT OF OFFICE / DELEGATION\n"
        "I am on medical leave (knee surgery, recovery through 21 Oct). Approval of "
        "the Q3 cutoff adjustment is delegated to Raj Venkatesan. Reach me at home, "
        "214 Alder Lane, Tacoma, WA 98402, only for urgent revenue recognition "
        "questions.",
        _labels(
            "redact",
            2,
            difficulty="routine",
            risk="medium",
            pii=(
                "knee surgery, recovery through 21 Oct",
                "214 Alder Lane, Tacoma, WA 98402",
            ),
            tools=("apply_redactions",),
        ),
    ),
    _page(
        "ARC-000472",
        "Nadia Brooks",
        "Tomas Brandt",
        "ATTORNEY-CLIENT PRIVILEGED -- PREPARED AT THE DIRECTION OF COUNSEL\n"
        "From: Tomas Brandt, General Counsel\n"
        "To: Audit Committee\n"
        "Memo on litigation exposure if the Q3 cutoff is restated. In our legal "
        "opinion the plaintiffs' strongest claim is that the 1 Oct shipments were "
        "recognized early on purpose. We recommend the Committee not minute any "
        "discussion of intent.",
        _labels("attorney_review", 2, risk="high", privileged=True, tools=_PRIV_TOOLS),
        to=("Audit Committee",),
    ),
    _page(
        "ARC-000518",
        "Tomas Brandt",
        "Helen Park",
        "PRIVILEGED & CONFIDENTIAL -- ATTORNEY WORK PRODUCT\n"
        "From: Helen Park, Kessler & Moore LLP\n"
        "To: Tomas Brandt, General Counsel\n"
        "Draft response to plaintiffs' second set of requests on Q3 revenue "
        "recognition. Our strategy is to object to the cutoff-restatement request "
        "as overbroad and to offer the audit committee minutes instead.",
        _labels("attorney_review", 2, risk="high", privileged=True, tools=_PRIV_TOOLS),
        to=("Tomas Brandt",),
    ),
    _page(
        "ARC-000284",
        "Dana Whitfield",
        "Priya Raghunathan",
        "ARCADIA FREIGHT -- EMAIL\n"
        "From: Priya Raghunathan\n"
        "To: Dana Whitfield\n"
        "Dana -- following up on our call. Based on what you described, I would "
        "treat the $3.1M as a Q4 item and I would hold off writing down the "
        "rationale until I have spoken to Tomas about the cutoff.",
        _labels("attorney_review", 2, risk="high", privileged=True, tools=_PRIV_TOOLS),
        to=("Dana Whitfield",),
    ),
    _page(
        "ARC-000299",
        "Dana Whitfield",
        "Dana Whitfield",
        "ARCADIA FREIGHT -- EMAIL\n"
        "From: Dana Whitfield\n"
        "To: Priya Raghunathan\n"
        "Priya -- do we have to tell the auditors about the $3.1M cutoff adjustment "
        "before the restatement vote, or can it wait for the Q4 close? I need to "
        "know before Thursday's call.",
        _labels("attorney_review", 2, risk="high", privileged=True, tools=_PRIV_TOOLS),
        to=("Priya Raghunathan",),
    ),
    _page(
        "ARC-000441",
        "Marcus Oyelaran",
        "Marcus Oyelaran",
        "ARCADIA FREIGHT -- EMAIL\n"
        "From: Marcus Oyelaran\n"
        "To: Finance leadership\n"
        "Spoke with Tomas Brandt this morning. His view is that we should not call "
        "the Q3 cutoff change an error in any written summary, and that we keep "
        "the wording neutral until the restatement question is decided. Please "
        "follow that in the close memo.",
        _labels(
            "attorney_review",
            2,
            difficulty="borderline",
            risk="high",
            privileged=True,
            tools=("escalate_to_attorney",),
        ),
        to=("Finance leadership",),
    ),
    _page(
        "ARC-000306",
        "Dana Whitfield",
        "Yusuf Demir",
        "ARCADIA FREIGHT -- EMAIL\n"
        "From: Yusuf Demir\n"
        "To: Dana Whitfield\n"
        "Cc: Priya Raghunathan\n"
        "Attaching the Q3 cutoff schedule for the close pack. Totals tie to the "
        "general ledger at $48.2M recognized revenue. Copying Priya for visibility "
        "only; there are no questions for Legal.",
        _labels(
            "produce",
            2,
            difficulty="trap",
            risk="medium",
            tools=("lookup_custodian",),
        ),
        to=("Dana Whitfield",),
        cc=("Priya Raghunathan",),
    ),
    _page(
        "ARC-000413",
        "Marcus Oyelaran",
        "Raj Venkatesan",
        "ARCADIA FREIGHT -- EMAIL\n"
        "From: Raj Venkatesan\n"
        "To: Marcus Oyelaran\n"
        "Cc: Tomas Brandt\n"
        "Updated timing for the $3.1M shipment revenue: goods departed 1 Oct and "
        "were recognized in Q3 under the old cutoff convention. Schedule attached. "
        "Legal is on the standing distribution list for close materials.",
        _labels(
            "produce",
            2,
            difficulty="trap",
            risk="medium",
            tools=("lookup_custodian",),
        ),
        to=("Marcus Oyelaran",),
        cc=("Tomas Brandt",),
    ),
    _page(
        "ARC-000329",
        "Marcus Oyelaran",
        "Marcus Oyelaran",
        "ARCADIA FREIGHT -- EMAIL\n"
        "From: Marcus Oyelaran\n"
        "To: Dana Whitfield, Raj Venkatesan\n"
        "If we restate Q3, the audit committee will want the effect on the lender "
        "covenant calculation. Can Treasury model Q3 EBITDA lower by $3.1M? We "
        "would be exposed to a covenant waiver request and I would rather know "
        "the number before the committee asks.",
        _labels("produce", 2, difficulty="trap", risk="medium"),
        to=("Dana Whitfield", "Raj Venkatesan"),
    ),
    _page(
        "ARC-000497",
        "Nadia Brooks",
        "Nadia Brooks",
        "ARCADIA FREIGHT -- EMAIL\n"
        "From: Nadia Brooks, CFO\n"
        "To: Lena Ortiz\n"
        "We will restate the Q3 revenue cutoff if the auditors insist. The "
        "regulatory exposure is real, so prepare the committee materials as if a "
        "comment letter on the quarterly filing is coming.",
        _labels("produce", 2, difficulty="trap", risk="medium"),
        to=("Lena Ortiz",),
    ),
    _page(
        "ARC-000467",
        "Marcus Oyelaran",
        "Unknown",
        "ARCADIA FREIGHT -- FAX SCAN (OCR, LOW CONFIDENCE)\n"
        "~~ Q3 c/o ... 3.1M ... [smudged] ... PRIV?GED ... fwd to T.B. "
        "[illegible] before the vote ~~ ... do not [illegible] to finance",
        _labels(
            "attorney_review",
            1,
            difficulty="borderline",
            risk="high",
            tools=("escalate_to_attorney",),
        ),
    ),
    _page(
        "ARC-000520",
        "Yusuf Demir",
        "Unknown",
        "ARCADIA FREIGHT -- SPREADSHEET SCAN (OCR, PARTIAL)\n"
        "Rev rec | cutoff t/o | Q3 | 3,1 0 0 ,O00 | ?? | chk w/ P.R. | [cut off]",
        _labels(
            "attorney_review",
            1,
            difficulty="borderline",
            risk="medium",
            tools=("escalate_to_attorney",),
        ),
    ),
    _page(
        "ARC-000259",
        "Grace Lindqvist",
        "Grace Lindqvist",
        "ARCADIA FREIGHT -- BENEFITS ENROLLMENT FORM\n"
        "Employee: Owen Fitch. SSN 900-55-1234. Dependent date of birth "
        "2016-04-09. Plan elected: family dental and vision. Effective 1 January.",
        _labels(
            "withhold",
            0,
            risk="medium",
            pii=("900-55-1234", "2016-04-09"),
        ),
    ),
    _page(
        "ARC-000333",
        "Ines Calloway",
        "Ines Calloway",
        "ARCADIA FREIGHT -- DOCK 2 VISITOR LOG\n"
        "Pedro Alvarez, driver license D-4410-8832, cell 555-0166, delivered pallets "
        "to Dock 2 at 07:40 and left at 08:05. Badge returned at the gate.",
        _labels(
            "withhold",
            0,
            risk="medium",
            pii=("D-4410-8832", "555-0166"),
        ),
    ),
    _page(
        "ARC-000572",
        "Raj Venkatesan",
        "Finance Reporting",
        "ARCADIA FREIGHT -- Q3 2024 REVENUE SUMMARY (INTERNAL) -- COPY\n"
        "Recognized revenue for the quarter ended 30 Sep 2024 was $48.2M, up 6% "
        "sequentially. $3.1M of this relates to shipments that departed on "
        "1 Oct and was recognized in Q3 under the prior cutoff convention. "
        "Finance is reviewing whether the cutoff should be restated.",
        _labels(
            "withhold",
            2,
            difficulty="trap",
            tools=("check_prior_production",),
        ),
    ),
    _page(
        "ARC-000580",
        "Raj Venkatesan",
        "Priya Raghunathan",
        "FW: PRIVILEGED AND CONFIDENTIAL -- ATTORNEY-CLIENT COMMUNICATION\n"
        "From: Priya Raghunathan, Associate General Counsel\n"
        "To: Marcus Oyelaran, VP Finance\n"
        "Marcus -- my advice is that the Q3 cutoff treatment is defensible but "
        "not free of risk. If we restate, we should expect the plaintiffs to "
        "argue the original recognition was knowing. Do not circulate this note "
        "outside the finance leadership group. [forwarded to Raj Venkatesan]",
        _labels(
            "attorney_review",
            2,
            difficulty="trap",
            risk="high",
            privileged=True,
            tools=("check_prior_production", "escalate_to_attorney"),
        ),
        to=("Marcus Oyelaran",),
    ),
)


_SHIPMENT_SCHEDULE = "\n".join(
    f"Line {n:02d}: shipment SHP-{4100 + 7 * n} departed {1 + n % 3} Oct, "
    f"invoice INV-{9200 + 11 * n}, amount ${(n * 37 + 120) * 1000:,}, "
    f"recognized in Q3 pending the cutoff test."
    for n in range(1, 39)
)


_LADING_SCHEDULE = "\n".join(
    f"Item {n:02d}: bill of lading BOL-{7300 + 13 * n} for customer lot "
    f"LOT-{510 + 17 * n}, value ${(n * 53 + 90) * 1000:,}, delivery confirmed "
    f"{30 - n % 5} Sep, cutoff test result: {'exception' if n % 4 == 0 else 'clear'}."
    for n in range(1, 35)
)


_EXTRA_PAGES = (
    _page(
        "ARC-000604",
        "Yusuf Demir",
        "Yusuf Demir",
        "ARCADIA FREIGHT -- EMAIL\n"
        "From: Yusuf Demir\n"
        "To: Dana Whitfield\n"
        "Subject: Q3 deferred revenue roll-forward\n"
        "Dana -- the deferred revenue roll-forward for Q3 is attached. The $3.1M "
        "of 1 Oct shipments sits in deferred revenue after the cutoff entry. "
        "Supporting schedule is WP-Q3-CUTOFF-14.",
        _labels("produce", 2),
        to=("Dana Whitfield",),
    ),
    _page(
        "ARC-000607",
        "Dana Whitfield",
        "Dana Whitfield",
        "ARCADIA FREIGHT -- Q3 REVENUE RECONCILIATION\n"
        "Sub-ledger recognized revenue of $48.2M agrees to the general ledger. "
        "Reconciling item: $3.1M of shipments dated 1 Oct, reversed from Q3 by "
        "journal entry JE-2024-0931. Prepared by Dana Whitfield, Controller.",
        _labels("produce", 2),
    ),
    _page(
        "ARC-000611",
        "Raj Venkatesan",
        "Raj Venkatesan",
        "ARCADIA FREIGHT -- TREASURY NOTE\n"
        "Cash collected in October on the $3.1M of Q3-recognized shipments totals "
        "$2.2M so far. If the cutoff is restated, the receivable moves with the "
        "revenue to Q4. Aging report is WP-Q3-AR-07.",
        _labels("produce", 2),
    ),
    _page(
        "ARC-000615",
        "Dana Whitfield",
        "Marcus Oyelaran",
        "ARCADIA FREIGHT -- EMAIL\n"
        "From: Marcus Oyelaran\n"
        "To: Dana Whitfield\n"
        "Cc: Tomas Brandt\n"
        "Subject: Q3 cutoff schedule for the close pack\n"
        "Dana -- final cutoff schedule attached: $3.1M of 1 Oct shipments, Q3 "
        "recognized revenue $48.2M. Tomas is copied because he is on the standing "
        "close distribution list; there is nothing here for Legal to answer.",
        _labels(
            "produce",
            2,
            difficulty="trap",
            risk="medium",
            tools=("lookup_custodian",),
        ),
        to=("Dana Whitfield",),
        cc=("Tomas Brandt",),
    ),
    _page(
        "ARC-000618",
        "Raj Venkatesan",
        "Yusuf Demir",
        "ARCADIA FREIGHT -- EMAIL\n"
        "From: Yusuf Demir\n"
        "To: Raj Venkatesan\n"
        "Cc: Helen Park\n"
        "Updated shipment timing list for the $3.1M cutoff item, as requested for "
        "the auditors' sample. Q3 recognized revenue is $48.2M. Outside counsel "
        "is copied on every auditor mailing; no advice is being asked for.",
        _labels(
            "produce",
            2,
            difficulty="trap",
            risk="medium",
            tools=("lookup_custodian",),
        ),
        to=("Raj Venkatesan",),
        cc=("Helen Park",),
    ),
    _page(
        "ARC-000622",
        "Dana Whitfield",
        "Yusuf Demir",
        "ARCADIA FREIGHT -- PRIVILEGED & CONFIDENTIAL STAMP (CLOSE SCHEDULE)\n"
        "From: Yusuf Demir\n"
        "To: Dana Whitfield\n"
        "Q3 cutoff test results: 14 of 212 bills of lading dated 30 Sep to 3 Oct "
        "fail the test, totaling $3.1M. The header above is the department's "
        "default stamp; no lawyer has seen or commented on this schedule.",
        _labels(
            "produce",
            2,
            difficulty="trap",
            risk="medium",
            tools=("lookup_custodian",),
        ),
        to=("Dana Whitfield",),
    ),
    _page(
        "ARC-000626",
        "Lena Ortiz",
        "Lena Ortiz",
        "ARCADIA FREIGHT -- EMAIL THREAD\n"
        "From: Lena Ortiz\n"
        "To: Nadia Brooks\n"
        "Subject: Committee packet and room booking\n"
        "Packet items: (1) the Q3 revenue cutoff memo quantifying the $3.1M "
        "timing item, (2) the restatement options table. Also, the boardroom is "
        "booked 14 November and lunch will be sandwiches.",
        _labels("produce", 2, difficulty="routine"),
        to=("Nadia Brooks",),
    ),
    _page(
        "ARC-000630",
        "Yusuf Demir",
        "Yusuf Demir",
        "ARCADIA FREIGHT -- Q3 CUTOFF TEST WORKPAPER WP-Q3-CUTOFF-15\n"
        "Population: bills of lading dated 30 Sep to 3 Oct. Recognized revenue "
        "under review: $48.2M. Exceptions total $3.1M.\n" + _SHIPMENT_SCHEDULE,
        _labels("produce", 2, difficulty="routine"),
    ),
    _page(
        "ARC-000633",
        "Lena Ortiz",
        "Lena Ortiz",
        "ARCADIA FREIGHT -- BOARD MINUTES (EXCERPT)\n"
        "The Board received the CFO's update on the Q3 revenue cutoff convention "
        "and the $3.1M timing item. The Board asked for the auditors' written "
        "view before any restatement decision. No vote was taken.",
        _labels("produce", 2),
    ),
    _page(
        "ARC-000637",
        "Ines Calloway",
        "Ines Calloway",
        "ARCADIA FREIGHT -- CAFETERIA NOTICE\n"
        "The cafeteria will close at 14:00 on Friday for deep cleaning. Hot meals "
        "resume Monday. Please return trays to the rack by the east door.",
        _labels("withhold", 0),
    ),
    _page(
        "ARC-000640",
        "Ines Calloway",
        "Ines Calloway",
        "ARCADIA FREIGHT -- PARKING NOTICE\n"
        "Level 2 of the staff garage is closed for line painting from Monday to "
        "Wednesday. Use Level 3 or the overflow lot behind Dock 1.",
        _labels("withhold", 0),
    ),
    _page(
        "ARC-000643",
        "Owen Fitch",
        "Owen Fitch",
        "ARCADIA FREIGHT -- WAGE GARNISHMENT ORDER (HR FILE)\n"
        "Employee: Pedro Alvarez, SSN 900-62-3841, home phone 555-0192. Garnish "
        "$220 per pay period starting 1 November, remit to the county clerk.",
        _labels(
            "withhold",
            0,
            difficulty="routine",
            risk="medium",
            pii=("900-62-3841", "555-0192"),
        ),
    ),
    _page(
        "ARC-000646",
        "Grace Lindqvist",
        "Grace Lindqvist",
        "ARCADIA FREIGHT -- LEAVE REQUEST FORM\n"
        "Employee: Wendell Okafor, date of birth 1988-02-17. Reason: outpatient "
        "treatment for a chronic condition, 3 days. Contact: 555-0121. Approved "
        "by the HR Business Partner.",
        _labels(
            "withhold",
            0,
            difficulty="routine",
            risk="medium",
            pii=(
                "1988-02-17",
                "outpatient treatment for a chronic condition",
                "555-0121",
            ),
        ),
    ),
    _page(
        "ARC-000649",
        "Lena Ortiz",
        "Lena Ortiz",
        "ARCADIA FREIGHT -- AUDIT COMMITTEE MINUTES (EXCERPT) -- RESENT COPY\n"
        "The Committee discussed the Q3 revenue cutoff convention and asked "
        "management to prepare a memo quantifying the effect of recognizing the "
        "$3.1M in Q4 instead. No decision was taken. Next review: 14 November.",
        _labels(
            "withhold",
            2,
            difficulty="trap",
            tools=("check_prior_production",),
        ),
    ),
    _page(
        "ARC-000652",
        "Dana Whitfield",
        "Dana Whitfield",
        "ARCADIA FREIGHT -- EXPENSE REPORT\n"
        "Employee: Dana Whitfield, Controller. Team lunch for the finance group "
        "after the audit walkthrough: $186.40. Parking: $22.00. No client "
        "entertainment. Approved by Nadia Brooks.",
        _labels("withhold", 0, difficulty="trap"),
    ),
    _page(
        "ARC-000656",
        "Yusuf Demir",
        "Yusuf Demir",
        "ARCADIA FREIGHT -- EMAIL\n"
        "From: Yusuf Demir\n"
        "To: Dana Whitfield\n"
        "Cutoff test is done: $3.1M of 1 Oct shipments confirmed, recognized "
        "revenue $48.2M, schedule WP-Q3-CUTOFF-16. I am out Friday; text my "
        "personal mobile, 555-0134, if the auditors call about timing.",
        _labels(
            "redact",
            2,
            difficulty="routine",
            risk="medium",
            pii=("555-0134",),
            tools=("apply_redactions",),
        ),
        to=("Dana Whitfield",),
    ),
    _page(
        "ARC-000659",
        "Yusuf Demir",
        "Noor Haddad",
        "ARCADIA FREIGHT -- SALES OPS EMAIL\n"
        "Second affected customer lot: Harbor Tile, $0.9M, departed 1 Oct. "
        "Delivery contact is Marisol Vega, 555-0187, "
        "marisol.vega@harbor-tile.example, who confirmed receipt on 2 Oct.",
        _labels(
            "redact",
            2,
            risk="medium",
            pii=("555-0187", "marisol.vega@harbor-tile.example"),
            tools=("apply_redactions",),
        ),
    ),
    _page(
        "ARC-000662",
        "Owen Fitch",
        "Owen Fitch",
        "ARCADIA FREIGHT -- PAYROLL ACCRUAL NOTE\n"
        "The Q3 sales bonus pool depends on recognized Q3 revenue; the $3.1M "
        "cutoff item decides the threshold. Pay the regional lead's accrual of "
        "$8,400 to payroll account 7710-336-02 only after the cutoff is settled.",
        _labels(
            "redact",
            2,
            risk="medium",
            pii=("7710-336-02",),
            tools=("apply_redactions",),
        ),
    ),
    _page(
        "ARC-000665",
        "Yusuf Demir",
        "Yusuf Demir",
        "ARCADIA FREIGHT -- Q3 CUTOFF WORKPAPER WP-Q3-CUTOFF-17\n"
        "Population: bills of lading dated 30 Sep to 3 Oct. Exceptions total "
        "$3.1M of $48.2M recognized revenue.\n" + _LADING_SCHEDULE + "\n"
        "Reviewer note: customer refund contact is Elena Marsh, 555-0155, "
        "elena.marsh@lakeside-mills.example.",
        _labels(
            "redact",
            2,
            difficulty="borderline",
            risk="medium",
            pii=("555-0155", "elena.marsh@lakeside-mills.example"),
            tools=("apply_redactions",),
        ),
    ),
    _page(
        "ARC-000668",
        "Marcus Oyelaran",
        "Marcus Oyelaran",
        "ARCADIA FREIGHT -- EMAIL THREAD\n"
        "From: Marcus Oyelaran\n"
        "To: Nadia Brooks\n"
        "Re: Q3 cutoff timing and Raj's absence\n"
        "Treasury will model the $3.1M shift to Q4 while Raj is out (surgery, "
        "employee ID 55210, back after 4 November). The Q3 recognized revenue "
        "figure of $48.2M does not change until the committee decides.",
        _labels(
            "redact",
            2,
            difficulty="routine",
            risk="medium",
            pii=("surgery", "55210"),
            tools=("apply_redactions",),
        ),
        to=("Nadia Brooks",),
    ),
    _page(
        "ARC-000672",
        "Nadia Brooks",
        "Tomas Brandt",
        "ARCADIA FREIGHT -- ATTORNEY-CLIENT PRIVILEGED LEGAL ADVICE\n"
        "From: Tomas Brandt, General Counsel\n"
        "To: Nadia Brooks, CFO\n"
        "Nadia -- legal advice on the Q3 cutoff: do not describe the $3.1M item "
        "as an error in the committee packet. If asked, say the convention was "
        "under review. We should talk before the vote.",
        _labels("attorney_review", 2, risk="high", privileged=True, tools=_PRIV_TOOLS),
        to=("Nadia Brooks",),
    ),
    _page(
        "ARC-000675",
        "Priya Raghunathan",
        "Helen Park",
        "ARCADIA FREIGHT -- PRIVILEGED ATTORNEY WORK PRODUCT\n"
        "From: Helen Park, Kessler & Moore LLP\n"
        "To: Priya Raghunathan, Associate General Counsel\n"
        "Our litigation plan on the cutoff claims: move to dismiss the intent "
        "theory, and prepare the Controller for deposition on the Q3 revenue "
        "recognition timeline.",
        _labels("attorney_review", 2, risk="high", privileged=True, tools=_PRIV_TOOLS),
        to=("Priya Raghunathan",),
    ),
    _page(
        "ARC-000678",
        "Raj Venkatesan",
        "Raj Venkatesan",
        "ARCADIA FREIGHT -- EMAIL\n"
        "From: Raj Venkatesan\n"
        "To: Tomas Brandt\n"
        "Tomas -- may we tell the lender about the $3.1M cutoff adjustment before "
        "the auditors conclude? I need your legal view on the covenant notice "
        "wording before Monday.",
        _labels("attorney_review", 2, risk="high", privileged=True, tools=_PRIV_TOOLS),
        to=("Tomas Brandt",),
    ),
    _page(
        "ARC-000681",
        "Dana Whitfield",
        "Dana Whitfield",
        "ARCADIA FREIGHT -- EMAIL\n"
        "From: Dana Whitfield\n"
        "To: Yusuf Demir, Raj Venkatesan\n"
        "Passing on what Priya told me after the committee call: she thinks we "
        "should book the $3.1M in Q4 and keep the memo short. Please draft the "
        "journal entry that way and do not attach her name.",
        _labels(
            "attorney_review",
            2,
            difficulty="borderline",
            risk="high",
            privileged=True,
            tools=("escalate_to_attorney",),
        ),
        to=("Yusuf Demir", "Raj Venkatesan"),
    ),
    _page(
        "ARC-000684",
        "Nadia Brooks",
        "Nadia Brooks",
        "ARCADIA FREIGHT -- EMAIL\n"
        "From: Nadia Brooks\n"
        "To: Lena Ortiz\n"
        "Our lawyers have looked at the cutoff problem and their view is that "
        "restating Q3 would hurt us in the lawsuit more than waiting. Keep that "
        "reasoning out of the committee minutes.",
        _labels(
            "attorney_review",
            2,
            difficulty="borderline",
            risk="high",
            privileged=True,
            tools=("escalate_to_attorney",),
        ),
        to=("Lena Ortiz",),
    ),
    _page(
        "ARC-000687",
        "Lena Ortiz",
        "Tomas Brandt",
        "ARCADIA FREIGHT -- FW: PRIVILEGED LEGAL MEMO\n"
        "From: Tomas Brandt, General Counsel\n"
        "To: Audit Committee\n"
        "Counsel's note on the auditors' request for our Q3 working files. As "
        "your lawyer I advise releasing only the cutoff schedule, and holding the "
        "internal emails until we have agreed a privilege log. [forwarded to "
        "Lena Ortiz]",
        _labels(
            "attorney_review",
            2,
            difficulty="trap",
            risk="high",
            privileged=True,
            tools=("lookup_custodian", "escalate_to_attorney"),
        ),
        to=("Audit Committee",),
    ),
    _page(
        "ARC-000690",
        "Yusuf Demir",
        "Unknown",
        "ARCADIA FREIGHT -- SCANNED NOTE (OCR, PARTIAL)\n"
        "[torn] ...Q3 ...cut off ... 3.1M ??? ... ask legal b4 [illegible] "
        "... not in the close pack ...",
        _labels(
            "attorney_review",
            1,
            difficulty="borderline",
            risk="medium",
            tools=("escalate_to_attorney",),
        ),
    ),
    _page(
        "ARC-000693",
        "Dana Whitfield",
        "Unknown",
        "ARCADIA FREIGHT -- HANDWRITTEN MARGIN NOTE (OCR, BLOTTED)\n"
        "bk rev?? ... 3 l M shipmts ... [blotted] ... call Tomas b4 signing "
        "... [illegible] ... [bottom of page missing]",
        _labels(
            "attorney_review",
            1,
            difficulty="borderline",
            risk="high",
            tools=("escalate_to_attorney",),
        ),
    ),
    _page(
        "ARC-000696",
        "Raj Venkatesan",
        "Raj Venkatesan",
        "ARCADIA FREIGHT -- EMAIL PRINTOUT (OCR, PARTIAL)\n"
        "From: Raj Venkatesan\n"
        "To: Marcus Oyelaran\n"
        "Freight rates for the Northpoint Mills lane went up 4% this quarter, so "
        "margin on that account is thinner. [smudged] ... whether the timing of "
        "their Oct[illegible] shipments ... [illegible] ... [rest of page cut off]",
        _labels(
            "attorney_review",
            1,
            difficulty="borderline",
            risk="medium",
            tools=("escalate_to_attorney",),
        ),
        to=("Marcus Oyelaran",),
    ),
    _page(
        "ARC-000699",
        "Marcus Oyelaran",
        "Marcus Oyelaran",
        "ARCADIA FREIGHT -- EMAIL THREAD\n"
        "From: Marcus Oyelaran\n"
        "To: Dana Whitfield\n"
        "Dana -- please add the Q3 cutoff schedule ($3.1M, recognized revenue "
        "$48.2M) to the close pack.\n"
        "---- Original message ----\n"
        "From: Priya Raghunathan\n"
        "To: Marcus Oyelaran\n"
        "My advice: do not describe the cutoff change as an error in any "
        "written summary until I have spoken to Tomas.",
        _labels(
            "attorney_review",
            2,
            difficulty="trap",
            risk="high",
            privileged=True,
            tools=("escalate_to_attorney",),
        ),
        to=("Dana Whitfield",),
    ),
)


EVAL_SET: tuple[ReviewPage, ...] = tuple(
    sorted(_GIST_PAGES + _NEW_PAGES + _EXTRA_PAGES, key=lambda page: page.doc_id)
)


def split_pages(
    pages: tuple[ReviewPage, ...],
) -> tuple[tuple[ReviewPage, ...], tuple[ReviewPage, ...]]:
    calibration: list[ReviewPage] = []
    holdout: list[ReviewPage] = []
    routes = sorted({page.labels.expected_route for page in pages})
    for route in routes:
        stratum = sorted(
            (p for p in pages if p.labels.expected_route == route),
            key=lambda page: page.doc_id,
        )
        calibration.extend(stratum[0::2])
        holdout.extend(stratum[1::2])
    by_id = lambda page: page.doc_id  # noqa: E731
    return tuple(sorted(calibration, key=by_id)), tuple(sorted(holdout, key=by_id))


CALIBRATION, HOLDOUT = split_pages(EVAL_SET)


REVIEW_APPENDIX = (
    "Internal appendix: legacy coding panels, vendor export mappings, custodian "
    "hold history, OCR confidence tables, outdated review macros, and audit "
    "metadata. "
) * 45


PRIOR_PRODUCTION: Mapping[str, Mapping[str, Any]] = {
    "ARC-000649": {
        "matched_doc_id": "ARC-000508",
        "similarity": 0.98,
        "status": "produced",
        "volume": "PROD-001",
    },
    "ARC-000572": {
        "matched_doc_id": "ARC-000141",
        "similarity": 0.97,
        "status": "produced",
        "volume": "PROD-001",
    },
    "ARC-000580": {
        "matched_doc_id": "ARC-000401",
        "similarity": 0.96,
        "status": "escalated_privileged",
        "volume": None,
    },
}
