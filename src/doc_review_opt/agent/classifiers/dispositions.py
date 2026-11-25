"""Parse an agent's reply into a route; shared by the simulation and the live loop."""

from __future__ import annotations

import re
from typing import NamedTuple

from pydantic import BaseModel, ValidationError

from doc_review_opt.models import Disposition, Route

FREE_TEXT_FORMAT_HINT = (
    "End your reply with exactly one line: "
    "FINAL DISPOSITION: <produce|withhold|attorney_review>; redact: <yes|no>"
)
_FREE_TEXT = re.compile(
    r"FINAL DISPOSITION:\s*(produce|withhold|attorney_review)\b"
    r"(?:\W+redact(?:ion)?:\s*(yes|no|true|false))?",
    re.IGNORECASE,
)
_YES = frozenset({"yes", "true"})


class DispositionReply(BaseModel):
    disposition: Disposition
    redact: bool
    reason: str


class ParsedDisposition(NamedTuple):
    disposition: Disposition
    redact: bool | None

    @property
    def route(self) -> Route:
        if self.disposition == "produce" and self.redact is not False:
            return "redact"
        return self.disposition


def parse_free_text(text: str) -> ParsedDisposition | None:
    matches = _FREE_TEXT.findall(text)
    if not matches:
        return None
    disposition, redact = matches[-1]
    chosen: Disposition = disposition.lower()
    return ParsedDisposition(chosen, redact.lower() in _YES if redact else None)


def parse_json_disposition(text: str) -> ParsedDisposition | None:
    try:
        reply = DispositionReply.model_validate_json(text)
    except ValidationError:
        return None
    return ParsedDisposition(reply.disposition, reply.redact)
