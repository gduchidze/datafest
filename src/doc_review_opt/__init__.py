"""Optimizing a document-review agent for cost and quality: Decisions API vs LLM."""

from doc_review_opt.agent.classifiers.decisions import (
    DecisionsClassifier,
    build_decisions_request,
)
from doc_review_opt.agent.classifiers.llm_probability import LLMProbabilityClassifier
from doc_review_opt.agent.graph import (
    BatchOutcome,
    build_review_graph,
    review_checkpointer,
    run_batch,
)
from doc_review_opt.agent.protocols import Classifier, Redactor
from doc_review_opt.agent.redactor import OpenAIRedactor
from doc_review_opt.agent.routing import RoutingPolicy, decide, route
from doc_review_opt.models import (
    ClassifierResult,
    PageLabels,
    PageResult,
    RedactionResult,
    ReviewPage,
    TokenUsage,
)
from doc_review_opt.pricing import approx_tokens, estimate_cost_usd

__all__ = [
    "BatchOutcome",
    "Classifier",
    "ClassifierResult",
    "DecisionsClassifier",
    "LLMProbabilityClassifier",
    "OpenAIRedactor",
    "PageLabels",
    "PageResult",
    "RedactionResult",
    "Redactor",
    "ReviewPage",
    "RoutingPolicy",
    "TokenUsage",
    "approx_tokens",
    "build_decisions_request",
    "build_review_graph",
    "decide",
    "estimate_cost_usd",
    "review_checkpointer",
    "route",
    "run_batch",
]
