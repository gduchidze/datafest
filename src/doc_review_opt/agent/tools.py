"""Local review tools the agent can call, and their function schemas."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from doc_review_opt.fixtures import (
    CUSTODIANS,
    PRIOR_PRODUCTION,
    PROTOCOL_SECTIONS,
    REVIEW_APPENDIX,
)


def lookup_custodian(name: str, payload: str = "slim") -> dict[str, Any]:
    custodian = CUSTODIANS.get(name)
    if custodian is None:
        return {"found": False, "name": name}
    if payload == "verbose":
        return {
            "found": True,
            "profile": custodian.__dict__,
            "directory": {n: c.__dict__ for n, c in CUSTODIANS.items()},
            "hold_audit_log": REVIEW_APPENDIX,
        }
    return {"found": True, "name": name, "is_attorney": custodian.is_attorney}


def lookup_protocol(section: str, payload: str = "slim") -> dict[str, Any]:
    body = PROTOCOL_SECTIONS.get(section)
    if body is None:
        return {"found": False, "section": section}
    if payload == "verbose":
        return {
            "found": True,
            "section": section,
            "text": body,
            "appendix": REVIEW_APPENDIX,
            "all_sections": PROTOCOL_SECTIONS,
        }
    return {"found": True, "section": section, "text": body}


def check_prior_production(doc_id: str, payload: str = "slim") -> dict[str, Any]:
    match = PRIOR_PRODUCTION.get(doc_id)
    if match is None:
        return {"found": False, "doc_id": doc_id}
    if payload == "verbose":
        return {
            "found": True,
            "doc_id": doc_id,
            **match,
            "index_dump": REVIEW_APPENDIX,
            "all_matches": PRIOR_PRODUCTION,
        }
    return {"found": True, "doc_id": doc_id, **match}


def apply_redactions(doc_id: str, spans: list[str]) -> dict[str, Any]:
    return {"doc_id": doc_id, "status": "applied", "spans_redacted": len(spans)}


def escalate_to_attorney(doc_id: str, reason: str) -> dict[str, Any]:
    return {"doc_id": doc_id, "queue": "attorney_review", "reason": reason}


class UnknownToolError(KeyError): ...


_TOOL_FUNCTIONS: Mapping[str, Callable[..., dict[str, Any]]] = {
    "lookup_custodian": lookup_custodian,
    "lookup_protocol": lookup_protocol,
    "check_prior_production": check_prior_production,
    "apply_redactions": apply_redactions,
    "escalate_to_attorney": escalate_to_attorney,
}


_LOOKUP_TOOLS = frozenset(
    {"lookup_custodian", "lookup_protocol", "check_prior_production"}
)


def execute_local_tool(
    name: str, args: Mapping[str, Any], payload: str = "slim"
) -> dict[str, Any]:
    function = _TOOL_FUNCTIONS.get(name)
    if function is None:
        raise UnknownToolError(name)
    if name in _LOOKUP_TOOLS:
        return function(**args, payload=payload)
    return function(**args)


def function_schema(
    name: str, description: str, properties: dict[str, Any], required: list[str]
) -> dict[str, Any]:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
        "strict": True,
    }


VERBOSE_TOOLS: list[dict[str, Any]] = [
    function_schema(
        "lookup_custodian",
        "Retrieve the complete custodian profile, the full directory, hold history, "
        "audit metadata, and any internal notes that could possibly help decide "
        "whether a person is an attorney.",
        {
            "name": {
                "type": "string",
                "description": "The full name of the author, recipient or cc.",
            }
        },
        ["name"],
    ),
    function_schema(
        "lookup_protocol",
        "Retrieve the complete protocol section, all appendix material, coding "
        "macros, exception examples, and neighbouring sections.",
        {
            "section": {
                "type": "string",
                "enum": list(PROTOCOL_SECTIONS),
                "description": "Protocol section to retrieve.",
            }
        },
        ["section"],
    ),
    function_schema(
        "check_prior_production",
        "Retrieve the full prior production index entry, similarity diagnostics, "
        "volume history and every other match, for any page that might be a copy.",
        {"doc_id": {"type": "string", "description": "The control number."}},
        ["doc_id"],
    ),
    function_schema(
        "apply_redactions",
        "Apply redactions in the review platform. Use for any page that might "
        "contain personal information, even when unsure.",
        {
            "doc_id": {"type": "string"},
            "spans": {"type": "array", "items": {"type": "string"}},
        },
        ["doc_id", "spans"],
    ),
    function_schema(
        "escalate_to_attorney",
        "Escalate a page to an attorney when the page is ambiguous, possibly "
        "privileged, damaged, or otherwise risky, with a full explanation.",
        {"doc_id": {"type": "string"}, "reason": {"type": "string"}},
        ["doc_id", "reason"],
    ),
]


SLIM_TOOLS: list[dict[str, Any]] = [
    function_schema(
        "lookup_custodian",
        "Check whether a named person is an attorney.",
        {"name": {"type": "string"}},
        ["name"],
    ),
    function_schema(
        "lookup_protocol",
        "Fetch one protocol section.",
        {"section": {"type": "string", "enum": list(PROTOCOL_SECTIONS)}},
        ["section"],
    ),
    function_schema(
        "check_prior_production",
        "Check whether a page duplicates an earlier production.",
        {"doc_id": {"type": "string"}},
        ["doc_id"],
    ),
    function_schema(
        "apply_redactions",
        "Redact the listed personal-information spans from a page to be produced.",
        {
            "doc_id": {"type": "string"},
            "spans": {"type": "array", "items": {"type": "string"}},
        },
        ["doc_id", "spans"],
    ),
    function_schema(
        "escalate_to_attorney",
        "Send a page to an attorney; give the reason only, never quote the page.",
        {"doc_id": {"type": "string"}, "reason": {"type": "string"}},
        ["doc_id", "reason"],
    ),
]


def allowed_tool_choice(names: list[str]) -> dict[str, Any]:
    return {
        "type": "allowed_tools",
        "mode": "auto",
        "tools": [{"type": "function", "name": name} for name in names],
    }
