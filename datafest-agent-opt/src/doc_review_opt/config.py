"""Typed, validated settings loaded once from the packaged config.toml."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from functools import lru_cache
from importlib.resources import files
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from doc_review_opt.exceptions import ConfigError


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Models(_Frozen):
    decisions: str
    terra: str
    luna: str
    judge: str
    triage: str
    redaction: str
    second_opinion: str
    batch: str
    decisions_price_key: str


class Pricing(_Frozen):
    batch_discount: float
    cache_min_tokens: int


class PriceRow(_Frozen):
    input: float
    cached_input: float
    cache_write: float
    output: float


class PolicyValues(_Frozen):
    privilege_escalation: float
    pii_redaction: float
    not_responsive: float
    responsive: float
    responsive_confidence: float


class Policies(_Frozen):
    default: PolicyValues
    calibrated: PolicyValues


class Calibration(_Frozen):
    privilege_escalation: tuple[float, ...]
    pii_redaction: tuple[float, ...]
    not_responsive: tuple[float, ...]
    responsive: tuple[float, ...]
    responsive_confidence: tuple[float, ...]
    sweep_thresholds: tuple[float, ...]


class Live(_Frozen):
    page_limit: int
    budget_limit_usd: float
    max_concurrency: int
    per_page_ceiling_usd: float
    judge_call_ceiling_usd: float
    redaction_call_ceiling_usd: float
    timeout_s: float
    max_retries: int
    max_tool_rounds: int
    triage_max_output_tokens: int
    judge_max_output_tokens: int
    redaction_max_output_tokens: int
    llm_max_output_tokens: int
    reasoning_effort: Literal["none"]


class Batch(_Frozen):
    max_output_tokens: int
    background_output_tokens: int


class Attorney(_Frozen):
    review_cost_usd_per_page: float


class Quality(_Frozen):
    base: float
    max: float
    leak_penalty: float
    wrong_route_penalty: float
    escalation_mismatch_penalty: float
    responsive_withheld_penalty: float
    unnecessary_escalation_penalty: float
    parse_failure_penalty: float
    extra_tool_penalty: float
    extra_tool_penalty_cap: float
    max_extra_tools_before_flag: int
    snippet_chars: int
    guardrail_pass_floor: float


class Metrics(_Frozen):
    median_quantile: float = Field(gt=0, lt=1)
    tail_quantile: float = Field(gt=0, le=1)
    route_accuracy_floor: float = Field(ge=0, le=1)
    relevance_recall_floor: float = Field(ge=0, le=1)
    privileged_recall_floor: float = Field(ge=0, le=1)
    pii_recall_floor: float = Field(ge=0, le=1)
    max_under_escalation_rate: float = Field(ge=0, le=1)
    max_unnecessary_escalation_rate: float = Field(ge=0, le=1)


class EngineProfile(_Frozen):
    engine: Literal["decisions", "llm"]
    model: str
    peak: float
    trap_peak: float
    priv_simple: float
    priv_hard: float
    cc_privileged: float
    trap_privileged: float
    pii_routine: float
    decisive_confidence: tuple[float, float] | None = None


class Simulation(_Frozen):
    base_latency_s: float
    input_tokens_per_s: float
    output_tokens_per_s: float
    tool_latency_s: float
    triage_latency_s: float
    sync_background_latency_s: float
    cached_latency_fraction: float
    decisions_base_latency_s: float
    decisions_tokens_per_s: float
    decisions_question_latency_s: float
    triage_overhead_tokens: int
    triage_output_tokens: int
    decisions_output_tokens_per_question: int
    llm_output_tokens: int
    sync_background_output_tokens: int
    baseline_missed_redaction: tuple[str, ...]
    baseline_unparseable: str
    luna_slip: str
    jitter_amplitude: float
    priv_jitter_amplitude: float
    decisive_score: float
    borderline_probabilities: tuple[float, float, float]
    clear_nonprivileged: float
    cc_privilege_with_protocol_wording: float
    borderline_nonprivileged: float
    pii_clear: float
    pii_simple: float
    routine_peak_discount: float
    near_share: float
    band_confidence: tuple[float, float]
    clear_confidence: float
    min_peak: float
    max_peak: float
    model_latency_s: dict[str, float]
    engines: dict[str, EngineProfile]


class VariantConfig(_Frozen):
    engine: Literal["agent", "decisions"]
    cascade: bool
    model: str
    routine_model: str | None = None
    payload: Literal["verbose", "slim"]
    strict_json: bool = True
    all_tools: bool
    carried_context_tokens: int
    sync_background_tokens: int
    triage: bool
    cache: bool
    async_background: bool
    verbosity: Literal["low", "high"]
    max_output_tokens: int
    question_wording: Literal["gist", "protocol"] = "protocol"


class Scoring(_Frozen):
    quality_ok: int
    quality_bad: int
    policy_ok: int
    policy_bad: int
    latency_ok: int
    latency_bad: int
    budget_ok: int
    budget_bad: int
    async_ok: int
    async_bad: int
    escalation_ok: int
    escalation_bad: int
    cache_penalty_low: int
    cache_penalty_medium: int
    recommended_bonus: int
    monthly_pages: int


class Synthetic(_Frozen):
    base_variant: str
    split_variant: str
    latency_saving_s: float
    latency_floor_s: float
    sync_background_cost_usd: float


class ArchitectureOption(_Frozen):
    label: str
    variant: str
    models: str
    tools: str
    cache: str
    workflow: str
    best_for: str


class OperatingScenario(_Frozen):
    scenario: str
    description: str
    quality_floor: float
    policy_floor: float
    p50_latency_target_s: float
    monthly_budget_100k_pages_usd: float
    max_escalation_rate: float
    needs_async: bool
    cache_locality: Literal["low", "medium", "high"]
    recommended: tuple[str, ...]


class Palette(_Frozen):
    surface: str
    text: str
    grid: str
    neutral: str
    ink: str
    blue: str
    orange: str
    aqua: str
    yellow: str
    magenta: str
    green: str
    good: str
    critical: str


class Chart(_Frozen):
    width: float
    height: float
    tall_height: float
    font_size: float
    title_size: float
    line_width: float
    marker_size: float
    bar_width: float
    grid_width: float
    label_pad: float
    min_label_share: float
    log_floor_factor: float
    log_headroom: float
    label_box_pad: float
    threshold_axis_max: float


class LiveRound(_Frozen):
    variant: str
    quality: float
    escalation_rate: float
    sync_tokens: int
    p50_latency_s: float
    model_cost_per_page_usd: float


class LiveEngine(_Frozen):
    engine: str
    accuracy: float
    model_cost_per_page_usd: float
    p50_latency_s: float


class LiveRun(_Frozen):
    date: str
    pages: int
    cost_usd: float
    cheapest_engine: str
    rounds: tuple[LiveRound, ...]
    engines: tuple[LiveEngine, ...]


class Settings(_Frozen):
    models: Models
    pricing: Pricing
    prices: dict[str, PriceRow]
    policy: Policies
    calibration: Calibration
    live: Live
    batch: Batch
    attorney: Attorney
    quality: Quality
    metrics: Metrics
    simulation: Simulation
    variants: dict[str, VariantConfig]
    scoring: Scoring
    synthetic: Synthetic
    architecture: dict[str, ArchitectureOption]
    scenarios: tuple[OperatingScenario, ...]
    palette: Palette
    entity_colors: dict[str, str]
    chart: Chart
    live_run: LiveRun


def parse_settings(raw: Mapping[str, Any]) -> Settings:
    try:
        return Settings.model_validate(raw)
    except ValidationError as error:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in error.errors()
        )
        raise ConfigError(f"invalid config.toml: {problems}") from error


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    text = files("doc_review_opt").joinpath("config.toml").read_text("utf-8")
    return parse_settings(tomllib.loads(text))


class DiagramNode(_Frozen):
    id: str
    label: str
    col: float
    row: float
    kind: str


class DiagramEdge(_Frozen):
    source: str = Field(alias="from")
    target: str = Field(alias="to")
    label: str = ""
    style: Literal["solid", "dashed"] = "solid"
    at: float = 0.5


class DiagramGroup(_Frozen):
    label: str
    nodes: tuple[str, ...]


class DiagramSpec(_Frozen):
    title: str
    cols: float
    rows: float
    nodes: tuple[DiagramNode, ...]
    edges: tuple[DiagramEdge, ...]
    groups: tuple[DiagramGroup, ...] = ()


class KindStyle(_Frozen):
    fill: str
    text: str
    edge: str | None = None


class DiagramLayout(_Frozen):
    row_height: float
    top_margin: float
    bottom_margin: float
    pad_x: float
    pad_y: float
    min_gap: float
    group_pad: float
    group_label_pad: float
    font_size: float
    label_font_size: float
    title_size: float
    line_width: float
    model_line_width: float
    arrow_scale: float
    corner: float


class GraphNodeStyle(_Frozen):
    label: str
    kind: str


class GraphStyle(_Frozen):
    title: str
    group: str
    nodes: dict[str, GraphNodeStyle]


class DiagramSettings(_Frozen):
    layout: DiagramLayout
    kinds: dict[str, KindStyle]
    diagrams: dict[str, DiagramSpec]
    graph: GraphStyle


def _placeholders(settings: Settings) -> dict[str, str]:
    return {
        **{k: str(v) for k, v in settings.policy.calibrated.model_dump().items()},
        **settings.models.model_dump(),
        "batch_off": f"{1 - settings.pricing.batch_discount:.0%}",
        "guardrail_floor": f"{settings.quality.guardrail_pass_floor:.0%}",
    }


def _resolve(spec: DiagramSpec, values: Mapping[str, str]) -> DiagramSpec:
    ids = {node.id for node in spec.nodes}
    linked = {end for edge in spec.edges for end in (edge.source, edge.target)}
    linked |= {node for group in spec.groups for node in group.nodes}
    if linked - ids:
        raise ConfigError(f"diagram {spec.title!r} links unknown nodes {linked - ids}")
    try:
        nodes = tuple(
            node.model_copy(update={"label": node.label.format_map(values)})
            for node in spec.nodes
        )
        edges = tuple(
            edge.model_copy(update={"label": edge.label.format_map(values)})
            for edge in spec.edges
        )
        groups = tuple(
            group.model_copy(update={"label": group.label.format_map(values)})
            for group in spec.groups
        )
    except KeyError as error:
        raise ConfigError(
            f"diagram {spec.title!r}: unknown placeholder {error}"
        ) from error
    return spec.model_copy(update={"nodes": nodes, "edges": edges, "groups": groups})


def parse_diagrams(raw: Mapping[str, Any], settings: Settings) -> DiagramSettings:
    try:
        parsed = DiagramSettings.model_validate(raw)
    except ValidationError as error:
        raise ConfigError(f"invalid diagrams.toml: {error}") from error
    palette = set(settings.palette.model_dump())
    for kind, style in parsed.kinds.items():
        if not {style.fill, style.text, style.edge or style.text} <= palette:
            raise ConfigError(f"diagram kind {kind!r} uses a color not in [palette]")
    values = _placeholders(settings)
    diagrams = {name: _resolve(spec, values) for name, spec in parsed.diagrams.items()}
    used = {
        f"diagram {name!r}": {node.kind for node in spec.nodes}
        for name, spec in diagrams.items()
    }
    used["[graph]"] = {style.kind for style in parsed.graph.nodes.values()}
    for where, kinds in used.items():
        unknown = kinds - set(parsed.kinds)
        if unknown:
            raise ConfigError(f"{where} uses unknown kinds {unknown}")
    return parsed.model_copy(update={"diagrams": diagrams})


@lru_cache(maxsize=1)
def get_diagrams() -> DiagramSettings:
    text = files("doc_review_opt").joinpath("diagrams.toml").read_text("utf-8")
    return parse_diagrams(tomllib.loads(text), get_settings())
