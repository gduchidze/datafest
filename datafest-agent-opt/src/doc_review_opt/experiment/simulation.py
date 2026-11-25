"""Deterministic offline simulation of every round, run through the real graph."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

from doc_review_opt.agent.classifiers.cascade import (
    CascadeClassifier,
    PriorProductionGate,
)
from doc_review_opt.agent.classifiers.decisions import (
    QUESTIONS,
    Wording,
    build_decisions_request,
    page_block,
    protocol_block,
    questions,
)
from doc_review_opt.agent.classifiers.dispositions import (
    parse_free_text,
    parse_json_disposition,
)
from doc_review_opt.agent.classifiers.llm_probability import (
    classifier_rules,
    confidence,
)
from doc_review_opt.agent.classifiers.tool_agent import (
    agent_request,
    live_config_for_page,
)
from doc_review_opt.agent.prompts import REDACTION_PROMPT, STABLE_REVIEW_PREFIX
from doc_review_opt.agent.protocols import Classifier, Redactor
from doc_review_opt.agent.routing import RoutingPolicy
from doc_review_opt.agent.tools import execute_local_tool
from doc_review_opt.agent.variants import (
    ALL_TOOL_NAMES,
    BASELINE,
    EXTRA_VARIANTS,
    ROUTING,
    VARIANT_ORDER,
    VARIANTS,
    VariantSpec,
    default_policy,
)
from doc_review_opt.config import EngineProfile, get_settings
from doc_review_opt.experiment.live import CACHE_KEY
from doc_review_opt.experiment.runs import review_pages
from doc_review_opt.experiment.traces import Trace
from doc_review_opt.fixtures import CUSTODIANS, EVAL_SET, PROTOCOL, PROTOCOL_SECTIONS
from doc_review_opt.models import (
    Answer,
    ClassifierResult,
    PredicateAnswer,
    RedactionResult,
    ReviewPage,
    Route,
    ScoreAnswer,
    TokenUsage,
)
from doc_review_opt.pricing import (
    CACHE_MIN_TOKENS,
    DECISIONS_PRICE_KEY,
    approx_tokens,
    estimate_cost_usd,
)

SETTINGS = get_settings()
SIM = SETTINGS.simulation
MODELS = SETTINGS.models
PROFILES = SIM.engines
DECISIONS_QUESTIONS = len(QUESTIONS)
RESPONSIVE = 2

BASELINE_REPLY = (
    "I reviewed {doc_id} against the protocol, the custodian directory, the prior "
    "production index, the redaction system and the escalation queue, and I am "
    "including every detail so the attorneys can follow each step. Privilege log "
    "draft: author, recipients, date and the privilege asserted, described neutrally. "
    "QA: my own call looks consistent with the protocol on responsiveness, privilege "
    "and personal information. Production tags: confidentiality designation and "
    "volume assigned. {ending}"
)
UNPARSEABLE_ENDING = "My recommendation is that the team takes a careful look."


def scrub_pii(page: ReviewPage) -> str:
    text = page.text
    for span in page.labels.pii_spans:
        text = text.replace(span, "[REDACTED]")
    return text


def jitter(doc_id: str, engine: str, salt: str = "") -> float:
    digest = hashlib.sha256(f"{doc_id}|{engine}|{salt}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def responses_latency_s(
    model: str,
    usage: TokenUsage,
    *,
    tools: int = 0,
    triage: bool = False,
    sync_background: bool = False,
) -> float:
    effective_input = usage.input - usage.cached * (1 - SIM.cached_latency_fraction)
    return (
        SIM.base_latency_s
        + SIM.model_latency_s[model]
        + effective_input / SIM.input_tokens_per_s
        + usage.output / SIM.output_tokens_per_s
        + SIM.tool_latency_s * tools
        + (SIM.triage_latency_s if triage else 0)
        + (SIM.sync_background_latency_s if sync_background else 0)
    )


def decisions_latency_s(
    usage: TokenUsage, questions: int = DECISIONS_QUESTIONS
) -> float:
    effective_input = usage.input - usage.cached * (1 - SIM.cached_latency_fraction)
    return (
        SIM.decisions_base_latency_s
        + effective_input / SIM.decisions_tokens_per_s
        + SIM.decisions_question_latency_s * questions
    )


def _prefix_cache(
    enabled: bool, prefix_tokens: int, doc_id: str, primer: str | None
) -> tuple[int, int]:
    if not enabled or prefix_tokens < CACHE_MIN_TOKENS:
        return 0, 0
    return (0, prefix_tokens) if doc_id == primer else (prefix_tokens, 0)


def tool_arguments(name: str, page: ReviewPage) -> dict[str, Any]:
    known = next(
        (n for n in (page.author, *page.recipients, *page.cc) if n in CUSTODIANS),
        page.author,
    )
    arguments: dict[str, dict[str, Any]] = {
        "lookup_custodian": {"name": known},
        "lookup_protocol": {"section": next(iter(PROTOCOL_SECTIONS))},
        "check_prior_production": {"doc_id": page.doc_id},
        "apply_redactions": {
            "doc_id": page.doc_id,
            "spans": list(page.labels.pii_spans),
        },
        "escalate_to_attorney": {"doc_id": page.doc_id, "reason": "needs attorney"},
    }
    return arguments[name]


def _agent_model(spec: VariantSpec, page: ReviewPage) -> str:
    routine = spec.config.routine_model
    if routine and page.labels.risk != "high":
        return routine
    return spec.config.model


def _intended_route(
    spec: VariantSpec, page: ReviewPage, model: str
) -> tuple[Route, bool]:
    labels = page.labels
    expected = labels.expected_route
    if spec.variant == BASELINE:
        if page.doc_id == SIM.baseline_unparseable:
            return "attorney_review", False
        if expected == "produce" and labels.difficulty == "trap":
            return "attorney_review", True
        if expected == "redact" and page.doc_id in SIM.baseline_missed_redaction:
            return "produce", True
    slipped = model == spec.config.routine_model and page.doc_id == SIM.luna_slip
    if spec.variant == ROUTING and slipped:
        return "withhold", True
    return expected, True


def _reply_text(
    spec: VariantSpec, page: ReviewPage, route: Route, parseable: bool
) -> str:
    disposition, redact = ("produce", True) if route == "redact" else (route, False)
    if spec.config.strict_json:
        reply = {
            "disposition": disposition,
            "redact": redact,
            "reason": "protocol rule",
        }
        return json.dumps(reply)
    ending = f"FINAL DISPOSITION: {disposition}; redact: {'yes' if redact else 'no'}"
    ending = ending if parseable else UNPARSEABLE_ENDING
    return BASELINE_REPLY.format(doc_id=page.doc_id, ending=ending)


class SimulatedAgentClassifier:
    engine = "agent"

    def __init__(self, spec: VariantSpec) -> None:
        self._spec = spec

    async def classify(self, page: ReviewPage, _protocol: str, /) -> ClassifierResult:
        cfg = self._spec.config
        model = _agent_model(self._spec, page)
        tools = ALL_TOOL_NAMES if cfg.all_tools else list(page.labels.expected_tools)
        tool_tokens = sum(
            approx_tokens(execute_local_tool(n, tool_arguments(n, page), cfg.payload))
            for n in tools
        )
        route, parseable = _intended_route(self._spec, page, model)
        reply = _reply_text(self._spec, page, route, parseable)
        parse = parse_json_disposition if cfg.strict_json else parse_free_text
        parsed = parse(reply)
        main = TokenUsage(
            input=approx_tokens(self._spec.prompt)
            + approx_tokens(self._spec.tools)
            + approx_tokens(page_block(page))
            + cfg.carried_context_tokens
            + tool_tokens
            + cfg.sync_background_tokens,
            output=approx_tokens(reply),
        )
        triage = TokenUsage()
        if cfg.triage:
            triage = TokenUsage(
                input=approx_tokens(page_block(page)) + SIM.triage_overhead_tokens,
                output=SIM.triage_output_tokens,
            )
        return ClassifierResult(
            engine=self.engine,
            model=model,
            agent_disposition=parsed.route if parsed else "attorney_review",
            agent_redact=parsed.redact if parsed else None,
            usage=main + triage,
            latency_s=responses_latency_s(
                model,
                main,
                tools=len(tools),
                triage=cfg.triage,
                sync_background=cfg.sync_background_tokens > 0,
            ),
            cost_usd=estimate_cost_usd(model, main)
            + (estimate_cost_usd(MODELS.triage, triage) if cfg.triage else 0.0),
            models=(MODELS.triage, model) if cfg.triage else (model,),
            tools=tuple(tools),
            routing_tokens=triage.input + triage.output,
            parse_failed=parsed is None,
        )


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _responsive_probs(
    page: ReviewPage, name: str, profile: EngineProfile
) -> tuple[float, ...]:
    labels = page.labels
    if labels.responsive_level == 1:
        return SIM.borderline_probabilities
    if labels.difficulty == "trap":
        base = profile.trap_peak
    else:
        routine = labels.difficulty == "routine"
        base = profile.peak - (SIM.routine_peak_discount if routine else 0.0)
    wobble = SIM.jitter_amplitude * (2 * jitter(page.doc_id, name, "resp") - 1)
    peak = _clip(base + wobble, SIM.min_peak, SIM.max_peak)
    near = SIM.near_share * (1 - peak)
    far = 1 - peak - near
    if labels.responsive_level == RESPONSIVE:
        return (far, near, peak)
    return (peak, near, far)


def _confidence(
    page: ReviewPage,
    name: str,
    profile: EngineProfile,
    probs: tuple[float, ...],
    score: float,
) -> float:
    if profile.decisive_confidence is None:
        return confidence(list(probs))
    u = jitter(page.doc_id, name, "conf")
    if score >= SIM.decisive_score:
        low, high = profile.decisive_confidence
        return low + (high - low) * u
    if page.labels.responsive_level == 0:
        return SIM.clear_confidence
    low, span = SIM.band_confidence
    return low + span * u


def _privileged(
    page: ReviewPage, name: str, profile: EngineProfile, wording: Wording
) -> float:
    labels = page.labels
    if labels.privileged:
        hard = labels.difficulty in ("borderline", "trap")
        base = profile.priv_hard if hard else profile.priv_simple
    elif labels.difficulty == "trap":
        base = profile.cc_privileged if page.cc else profile.trap_privileged
        if page.cc and profile.engine == "decisions" and wording == "protocol":
            base = SIM.cc_privilege_with_protocol_wording
    elif labels.difficulty == "borderline":
        base = SIM.borderline_nonprivileged
    else:
        base = SIM.clear_nonprivileged
    wobble = SIM.priv_jitter_amplitude * (2 * jitter(page.doc_id, name, "priv") - 1)
    return _clip(base + wobble)


def _pii(page: ReviewPage, name: str, profile: EngineProfile) -> float:
    labels = page.labels
    if not labels.contains_pii:
        base = SIM.pii_clear
    else:
        base = profile.pii_routine if labels.difficulty == "routine" else SIM.pii_simple
    wobble = SIM.priv_jitter_amplitude * (2 * jitter(page.doc_id, name, "pii") - 1)
    return _clip(base + wobble)


def simulated_answers(
    page: ReviewPage, name: str, wording: Wording = "protocol"
) -> dict[str, Answer]:
    profile = PROFILES[name]
    probs = _responsive_probs(page, name, profile)
    score = sum(i * p for i, p in enumerate(probs))
    return {
        "responsive": ScoreAnswer(
            name="responsive",
            score=score,
            confidence=_confidence(page, name, profile, probs, score),
            probabilities=probs,
        ),
        "privileged": PredicateAnswer(
            name="privileged", probability=_privileged(page, name, profile, wording)
        ),
        "contains_pii": PredicateAnswer(
            name="contains_pii", probability=_pii(page, name, profile)
        ),
    }


class SimulatedTypedClassifier:
    def __init__(
        self,
        name: str,
        *,
        cache: bool = False,
        primer: str | None = None,
        sync_background_tokens: int = 0,
        wording: Wording = "protocol",
    ) -> None:
        self._name = name
        self._wording: Wording = wording
        self._profile = PROFILES[name]
        self._cache = cache
        self._primer = primer
        self._background = sync_background_tokens

    async def classify(self, page: ReviewPage, protocol: str) -> ClassifierResult:
        profile = self._profile
        prefix = approx_tokens(protocol_block(protocol))
        page_tokens = approx_tokens(page_block(page))
        if profile.engine == "decisions":
            cached, written = _prefix_cache(
                self._cache, prefix, page.doc_id, self._primer
            )
            usage = TokenUsage(
                input=prefix + page_tokens + approx_tokens(questions(self._wording)),
                cached=cached,
                cache_write=written,
                output=SIM.decisions_output_tokens_per_question * DECISIONS_QUESTIONS,
            )
            cost = estimate_cost_usd(DECISIONS_PRICE_KEY, usage)
            latency = decisions_latency_s(usage)
        else:
            usage = TokenUsage(
                input=approx_tokens(classifier_rules(self._wording))
                + prefix
                + page_tokens,
                output=SIM.llm_output_tokens,
            )
            cost = estimate_cost_usd(profile.model, usage)
            latency = responses_latency_s(profile.model, usage)
        models: tuple[str, ...] = (profile.model,)
        if self._background:
            extra = TokenUsage(
                input=page_tokens + self._background,
                output=SIM.sync_background_output_tokens,
            )
            usage, models = usage + extra, (*models, MODELS.luna)
            cost += estimate_cost_usd(MODELS.luna, extra)
            latency += responses_latency_s(MODELS.luna, extra)
        return ClassifierResult(
            engine=profile.engine,
            model=profile.model,
            answers=simulated_answers(page, self._name, self._wording),
            usage=usage,
            latency_s=latency,
            cost_usd=cost,
            models=tuple(dict.fromkeys(models)),
        )


class SimulatedRedactor:
    def __init__(self, *, cache: bool = False, primer: str | None = None) -> None:
        self._cache = cache
        self._primer = primer

    async def redact(self, page: ReviewPage) -> RedactionResult:
        text = scrub_pii(page)
        prefix = approx_tokens(REDACTION_PROMPT)
        if self._cache:
            prefix += approx_tokens(STABLE_REVIEW_PREFIX)
        cached, written = _prefix_cache(self._cache, prefix, page.doc_id, self._primer)
        usage = TokenUsage(
            input=prefix + approx_tokens(page.text),
            cached=cached,
            cache_write=written,
            output=approx_tokens(text),
        )
        model = MODELS.redaction
        return RedactionResult(
            text=text,
            model=model,
            usage=usage,
            latency_s=responses_latency_s(model, usage),
            cost_usd=estimate_cost_usd(model, usage),
        )


def build_simulated_engines(
    spec: VariantSpec, pages: Sequence[ReviewPage], policy: RoutingPolicy
) -> tuple[Classifier, Redactor]:
    cfg = spec.config
    if cfg.engine == "agent":
        return SimulatedAgentClassifier(spec), SimulatedRedactor()
    primer = min(p.doc_id for p in pages)
    pii_pages = [p.doc_id for p in pages if p.labels.pii_spans]
    classifier: Classifier = SimulatedTypedClassifier(
        "decisions",
        cache=cfg.cache,
        primer=primer,
        sync_background_tokens=cfg.sync_background_tokens,
        wording=cfg.question_wording,
    )
    if cfg.cascade:
        second = SimulatedTypedClassifier("llm-terra", wording=cfg.question_wording)
        classifier = CascadeClassifier(classifier, second, policy)
    redactor = SimulatedRedactor(cache=cfg.cache, primer=min(pii_pages, default=None))
    return PriorProductionGate(classifier), redactor


async def simulate_variant(
    variant: str,
    pages: Sequence[ReviewPage] = EVAL_SET,
    *,
    policy: RoutingPolicy | None = None,
) -> list[Trace]:
    spec = VARIANTS[variant]
    chosen = policy or default_policy(spec)
    classifier, redactor = build_simulated_engines(spec, pages, chosen)
    run = await review_pages(
        spec, pages, classifier, redactor, policy=chosen, mode="simulated"
    )
    return run.traces


async def simulate_all(
    pages: Sequence[ReviewPage] = EVAL_SET, *, include_extras: bool = False
) -> list[Trace]:
    variants = VARIANT_ORDER + (EXTRA_VARIANTS if include_extras else [])
    traces: list[Trace] = []
    for variant in variants:
        traces.extend(await simulate_variant(variant, pages))
    return traces


def variant_request_example(variant: str, page: ReviewPage) -> dict[str, Any]:
    spec = VARIANTS[variant]
    if spec.config.engine == "agent":
        return agent_request(live_config_for_page(page, variant), page)
    key = CACHE_KEY if spec.config.cache else None
    return build_decisions_request(page, PROTOCOL, cache_key=key)
