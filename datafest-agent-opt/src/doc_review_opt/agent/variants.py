"""The optimization rounds: config.toml numbers plus prompt and tool text."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from doc_review_opt.agent.prompts import (
    BAD_BASELINE_PROMPT,
    CONTROLLED_PROMPT,
    STABLE_REVIEW_PREFIX,
)
from doc_review_opt.agent.routing import DEFAULT_POLICY, RoutingPolicy
from doc_review_opt.agent.tools import SLIM_TOOLS, VERBOSE_TOOLS
from doc_review_opt.config import VariantConfig, get_settings

SETTINGS = get_settings()
BASELINE = "00_bad_baseline"
CONTROLS = "01_prompt_tool_context_controls"
ROUTING = "02_model_routing"
DECISION_MODEL = "03_decision_model"
ROUTER_ONLY = "03a_router_only"
GIST_WORDING = "03b_gist_wording"
ORIGINAL_WORDING_VARIANT = GIST_WORDING
CACHING = "04_prompt_caching"
SPLIT = "05_split_workflow"

VARIANT_ORDER = [BASELINE, CONTROLS, ROUTING, DECISION_MODEL, CACHING, SPLIT]
EXTRA_VARIANTS = [ROUTER_ONLY, GIST_WORDING]
VARIANT_LABELS = {
    BASELINE: "Bad baseline",
    CONTROLS: "Round 1: controls",
    ROUTING: "Round 2: routing",
    DECISION_MODEL: "Round 3: decision model",
    ROUTER_ONLY: "Decisions + code router",
    GIST_WORDING: "Round 3, original wording",
    CACHING: "Round 4: caching",
    SPLIT: "Round 5: split workflow",
}
ALL_TOOL_NAMES = [str(tool["name"]) for tool in SLIM_TOOLS]
CALIBRATED_POLICY = RoutingPolicy(**SETTINGS.policy.calibrated.model_dump())


@dataclass(frozen=True)
class VariantSpec:
    variant: str
    label: str
    prompt: str
    tools: list[dict[str, Any]]
    config: VariantConfig


def _text(variant: str) -> tuple[str, list[dict[str, Any]]]:
    if SETTINGS.variants[variant].engine == "decisions":
        return STABLE_REVIEW_PREFIX, SLIM_TOOLS
    if variant == BASELINE:
        return BAD_BASELINE_PROMPT, VERBOSE_TOOLS
    return CONTROLLED_PROMPT, SLIM_TOOLS


VARIANTS: dict[str, VariantSpec] = {
    variant: VariantSpec(variant, VARIANT_LABELS[variant], *_text(variant), config)
    for variant, config in SETTINGS.variants.items()
}


def default_policy(spec: VariantSpec) -> RoutingPolicy:
    return CALIBRATED_POLICY if spec.config.engine == "decisions" else DEFAULT_POLICY
