"""The tool-calling agent classifier: request builders, tool loop and triage."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any, Literal

from langsmith import traceable
from openai import AsyncOpenAI, OpenAIError
from openai.types.responses import Response
from pydantic import BaseModel, ConfigDict, ValidationError

from doc_review_opt.agent.classifiers.decisions import (
    page_block,
    trace_inputs,
)
from doc_review_opt.agent.classifiers.dispositions import (
    FREE_TEXT_FORMAT_HINT,
    parse_free_text,
    parse_json_disposition,
)
from doc_review_opt.agent.classifiers.responses import (
    completed_text,
    usage_from_response,
)
from doc_review_opt.agent.prompts import DISPOSITION_FORMAT, TRIAGE_INSTRUCTIONS
from doc_review_opt.agent.tools import (
    UnknownToolError,
    allowed_tool_choice,
    execute_local_tool,
)
from doc_review_opt.agent.variants import (
    BASELINE,
    VARIANTS,
    VariantSpec,
)
from doc_review_opt.config import get_settings
from doc_review_opt.exceptions import (
    BilledCallError,
    UnusableReplyError,
)
from doc_review_opt.models import (
    ClassifierResult,
    ReviewPage,
    TokenUsage,
)
from doc_review_opt.pricing import approx_tokens, estimate_cost_usd

logger = logging.getLogger(__name__)


SETTINGS = get_settings()


LIVE = SETTINGS.live


MODELS = SETTINGS.models


REASONING = {"effort": LIVE.reasoning_effort}


TOOL_ERRORS = (UnknownToolError, TypeError, ValueError, KeyError)


AGENT_ERRORS = (OpenAIError, UnusableReplyError, BilledCallError, ValidationError)


TRIAGE_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "name": "page_triage",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "responsive": {"type": "boolean"},
            "privileged": {"type": "boolean"},
            "pii": {"type": "boolean"},
            "needs_attorney": {"type": "boolean"},
        },
        "required": ["responsive", "privileged", "pii", "needs_attorney"],
        "additionalProperties": False,
    },
}


@dataclass(frozen=True)
class LiveReviewConfig:
    name: str
    model: str
    instructions: str
    tools: list[dict[str, Any]]
    payload: Literal["verbose", "slim"]
    verbosity: Literal["low", "high"]
    max_output_tokens: int
    tool_choice: str | dict[str, Any]
    strict_json: bool


def _agent_spec(variant: str) -> VariantSpec:
    spec = VARIANTS.get(variant)
    if spec is None or spec.config.engine != "agent":
        raise ValueError(f"{variant!r} is not an agent variant (00, 01 or 02)")
    return spec


def routed_model(spec: VariantSpec, risk: str) -> str:
    routine = spec.config.routine_model
    return routine if routine and risk != "high" else spec.config.model


def live_config_for_page(page: ReviewPage, variant: str) -> LiveReviewConfig:
    spec = _agent_spec(variant)
    names = list(page.labels.expected_tools)
    if variant == BASELINE:
        choice: str | dict[str, Any] = "required"
    else:
        choice = allowed_tool_choice(names) if names else "none"
    return LiveReviewConfig(
        name=variant,
        model=routed_model(spec, page.labels.risk),
        instructions=spec.prompt,
        tools=spec.tools,
        payload=spec.config.payload,
        verbosity=spec.config.verbosity,
        max_output_tokens=spec.config.max_output_tokens,
        tool_choice=choice,
        strict_json=spec.config.strict_json,
    )


def _user_message(page: ReviewPage, strict_json: bool) -> dict[str, str]:
    content = page_block(page)
    if not strict_json:
        content += "\n\n" + FREE_TEXT_FORMAT_HINT
    return {"role": "user", "content": content}


def agent_request(
    config: LiveReviewConfig,
    page: ReviewPage,
    input_items: list[Any] | None = None,
    *,
    tool_choice: str | dict[str, Any] | None = None,
) -> dict[str, Any]:
    text: dict[str, Any] = {"verbosity": config.verbosity}
    if config.strict_json:
        text["format"] = DISPOSITION_FORMAT
    return {
        "model": config.model,
        "instructions": config.instructions,
        "tools": config.tools,
        "input": input_items or [_user_message(page, config.strict_json)],
        "reasoning": REASONING,
        "text": text,
        "max_output_tokens": config.max_output_tokens,
        "parallel_tool_calls": True,
        "tool_choice": tool_choice or config.tool_choice,
    }


def triage_request(page: ReviewPage) -> dict[str, Any]:
    return {
        "model": MODELS.triage,
        "instructions": TRIAGE_INSTRUCTIONS,
        "input": page_block(page),
        "reasoning": REASONING,
        "text": {"format": TRIAGE_SCHEMA, "verbosity": "low"},
        "max_output_tokens": LIVE.triage_max_output_tokens,
    }


class _Triage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    responsive: bool
    privileged: bool
    pii: bool
    needs_attorney: bool

    @property
    def risk(self) -> str:
        if self.privileged or self.needs_attorney:
            return "high"
        return "medium" if self.pii else "low"


@dataclass(frozen=True)
class ToolLoop:
    response: Response
    usage: TokenUsage
    tools: tuple[str, ...]


def _function_calls(response: Response) -> list[Any]:
    return [i for i in response.output if getattr(i, "type", None) == "function_call"]


def _run_tool(call: Any, payload: str) -> dict[str, Any]:
    try:
        return execute_local_tool(
            call.name, json.loads(call.arguments or "{}"), payload
        )
    except TOOL_ERRORS as error:
        return {"error": f"{type(error).__name__}: {error}"}


def _next_request_cost(
    config: LiveReviewConfig, last: TokenUsage, tool_outputs: list[str]
) -> float:
    grown = last.input + last.output + sum(approx_tokens(o) for o in tool_outputs)
    worst = TokenUsage(input=grown, output=config.max_output_tokens)
    return estimate_cost_usd(config.model, worst)


def _execute_calls(
    calls: list[Any], payload: str
) -> tuple[list[str], list[dict[str, str]]]:
    names: list[str] = []
    outputs: list[dict[str, str]] = []
    for call in calls:
        names.append(call.name)
        outputs.append(
            {
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": json.dumps(_run_tool(call, payload)),
            }
        )
    return names, outputs


def _close_or_raise(
    config: LiveReviewConfig,
    usage: TokenUsage,
    last_usage: TokenUsage,
    outputs: list[dict[str, str]],
    ceiling_usd: float,
) -> bool:
    worst = _next_request_cost(config, last_usage, [o["output"] for o in outputs])
    spent = estimate_cost_usd(config.model, usage)
    if spent + worst > ceiling_usd:
        raise BilledCallError(
            f"the ${ceiling_usd:.3f} page ceiling leaves no room", usage, spent
        )
    return spent + 2 * worst > ceiling_usd


async def run_tool_loop(
    client: AsyncOpenAI,
    config: LiveReviewConfig,
    page: ReviewPage,
    max_tool_rounds: int = LIVE.max_tool_rounds,
    ceiling_usd: float = LIVE.per_page_ceiling_usd,
) -> ToolLoop:
    if max_tool_rounds < 1:
        raise ValueError("max_tool_rounds must be at least 1")
    items: list[Any] = [_user_message(page, config.strict_json)]
    usage = TokenUsage()
    names: list[str] = []

    def billed(message: str) -> BilledCallError:
        return BilledCallError(message, usage, estimate_cost_usd(config.model, usage))

    try:
        response = await client.responses.create(**agent_request(config, page))
        last_usage = usage = usage_from_response(response)
        for round_index in range(max_tool_rounds):
            calls = _function_calls(response)
            if not calls:
                break
            called, outputs = _execute_calls(calls, config.payload)
            names += called
            items += [*response.output, *outputs]
            closing = (
                _close_or_raise(config, usage, last_usage, outputs, ceiling_usd)
                or round_index + 1 == max_tool_rounds
            )
            request = agent_request(
                config, page, items, tool_choice="none" if closing else None
            )
            response = await client.responses.create(**request)
            last_usage = usage_from_response(response)
            usage = usage + last_usage
    except OpenAIError as error:
        raise billed(f"{type(error).__name__}: {error}") from error
    if _function_calls(response):
        raise billed("the final response still contains unexecuted tool calls")
    if response.status != "completed":
        raise billed(f"the response did not complete: status={response.status}")
    return ToolLoop(response, usage, tuple(names))


class LiveAgentClassifier:
    engine = "agent"

    def __init__(self, client: AsyncOpenAI, variant: str) -> None:
        self._client = client
        self._variant = variant
        self._spec = _agent_spec(variant)
        self._traced = traceable(
            run_type="chain",
            name=f"agent_review::{variant}",
            metadata={"engine": self.engine, "variant": variant},
            process_inputs=trace_inputs,
        )(self._classify)

    async def classify(self, page: ReviewPage, _protocol: str, /) -> ClassifierResult:
        result: ClassifierResult = await self._traced(page=page)
        return result

    async def _triage(self, page: ReviewPage) -> tuple[_Triage, TokenUsage]:
        response = await self._client.responses.create(**triage_request(page))
        usage = usage_from_response(response)
        try:
            triage = _Triage.model_validate_json(completed_text(response))
        except (UnusableReplyError, ValidationError) as error:
            cost = estimate_cost_usd(MODELS.triage, usage)
            raise BilledCallError(str(error), usage, cost) from error
        return triage, usage

    async def _routed_config(
        self, page: ReviewPage
    ) -> tuple[LiveReviewConfig, TokenUsage, float, tuple[str, ...]]:
        config = live_config_for_page(page, self._variant)
        if not self._spec.config.triage:
            return config, TokenUsage(), 0.0, ()
        triage, usage = await self._triage(page)
        cost = estimate_cost_usd(MODELS.triage, usage)
        model = routed_model(self._spec, triage.risk)
        return replace(config, model=model), usage, cost, (MODELS.triage,)

    def _failed(
        self,
        error: Exception,
        config: LiveReviewConfig,
        usage: TokenUsage,
        cost: float,
        latency_s: float,
    ) -> ClassifierResult:
        if isinstance(error, BilledCallError):
            usage, cost = usage + error.usage, cost + error.cost_usd
        return ClassifierResult(
            engine=self.engine,
            model=config.model,
            usage=usage,
            latency_s=latency_s,
            cost_usd=cost,
            error=f"{type(error).__name__}: {error}",
        )

    async def _classify(self, page: ReviewPage) -> ClassifierResult:
        started = perf_counter()
        config = live_config_for_page(page, self._variant)
        usage = TokenUsage()
        triage_cost = 0.0
        try:
            config, usage, triage_cost, models = await self._routed_config(page)
            loop = await run_tool_loop(self._client, config, page)
        except AGENT_ERRORS as error:
            logger.warning("agent call failed doc_id=%s: %s", page.doc_id, error)
            elapsed = perf_counter() - started
            return self._failed(error, config, usage, triage_cost, elapsed)
        parse = parse_json_disposition if config.strict_json else parse_free_text
        parsed = parse(loop.response.output_text)
        return ClassifierResult(
            engine=self.engine,
            model=config.model,
            agent_disposition=parsed.route if parsed else "attorney_review",
            agent_redact=parsed.redact if parsed else None,
            usage=usage + loop.usage,
            latency_s=perf_counter() - started,
            cost_usd=triage_cost + estimate_cost_usd(config.model, loop.usage),
            models=(*models, config.model),
            tools=loop.tools,
            routing_tokens=usage.input + usage.output,
            parse_failed=parsed is None,
        )


__all__ = [
    "TRIAGE_SCHEMA",
    "LiveAgentClassifier",
    "LiveReviewConfig",
    "agent_request",
    "live_config_for_page",
    "run_tool_loop",
    "triage_request",
]
