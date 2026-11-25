"""Prompt text for the review agent, the redactor and the triage call."""

from __future__ import annotations

import textwrap
from typing import Any

from doc_review_opt.fixtures import (
    PROTOCOL,
)

REDACTION_PROMPT = (
    "Return the page text with every piece of personal information replaced by the "
    "literal token [REDACTED]. Remove the underlying characters; do not mask, "
    "abbreviate, or summarize them. Preserve all other text exactly.\n\nPAGE:\n"
)


BAD_BASELINE_PROMPT = textwrap.dedent(
    """
    You are a comprehensive e-discovery review super-agent for a litigation team.

    Always be extremely thorough. Before deciding, gather as much context as possible
    from the protocol, the custodian directory, the prior production index, the
    redaction system and the escalation queue. Prefer using tools even when the
    answer may already be apparent. Explain your reasoning in detail so the
    reviewing attorneys understand every step.

    You are also responsible for drafting the privilege log entry, running QA on your
    own call, and writing production tags before you give your disposition.

    If there is any uncertainty, call more tools. If the protocol might be relevant,
    retrieve every section. If a person might be an attorney, look them up. If a page
    might be a copy, check prior production. If a page might contain personal
    information, apply redactions.
    """
).strip()


CONTROLLED_PROMPT = textwrap.dedent(
    """
    Role: First-pass document reviewer on matter 24-CV-118.

    Goal: Give each page exactly one disposition with the fewest necessary tool calls
    while never producing a privileged page or leaving personal information in
    produced text.

    Tool rules:
    - Clearly responsive or clearly unrelated page, no personal data: no tools.
    - Responsive page with personal data: apply_redactions with the exact spans.
    - Attorney named in From or To, or relayed legal advice: lookup_custodian if a
      name is unclear, then escalate_to_attorney with the reason only.
    - Attorney only in cc on business mail: lookup_custodian, then decide.
    - Possible copy of an earlier page: check_prior_production.
    - Damaged or illegible page: escalate_to_attorney.

    Output rules:
    - Return the disposition JSON only: disposition, redact, reason.
    - Never quote page text in the reason.
    """
).strip()


STABLE_REVIEW_PREFIX = (
    textwrap.dedent(
        """
        Role: First-pass document reviewer on matter 24-CV-118.
        Constraints: Be concise. One disposition per page: produce, withhold or
        attorney_review. Never produce a privileged page. Never quote page text.
        Output shape: disposition, redact, reason.
        Tool contract: tool definitions are stable across requests; restrict
        callable tools with tool_choice.allowed_tools.
        Version: document-review-optimization-v1.

        REVIEW PROTOCOL
        """
    ).strip()
    + "\n"
    + PROTOCOL
)


DISPOSITION_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "name": "review_disposition",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "disposition": {
                "type": "string",
                "enum": ["produce", "withhold", "attorney_review"],
            },
            "redact": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": ["disposition", "redact", "reason"],
        "additionalProperties": False,
    },
}


TRIAGE_INSTRUCTIONS = (
    "Triage one page for e-discovery review. Return only the structured object: "
    "whether the page is responsive to the matter, whether it is privileged (an "
    "attorney is author or recipient, or it relays legal advice), whether it holds "
    "personal information, and whether an attorney must see it."
)
