"""Frozen value types shared by every phase of the project."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Disposition = Literal["produce", "withhold", "attorney_review"]
Route = Disposition | Literal["redact"]
Difficulty = Literal["simple", "routine", "borderline", "trap"]
Risk = Literal["low", "medium", "high"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PageLabels(_Frozen):
    expected_disposition: Disposition
    expected_route: Route
    responsive_level: Literal[0, 1, 2]
    privileged: bool
    contains_pii: bool
    pii_spans: tuple[str, ...] = ()
    must_escalate: bool
    difficulty: Difficulty
    risk: Risk
    expected_tools: tuple[str, ...] = ()


class ReviewPage(_Frozen):
    doc_id: str
    custodian: str
    author: str
    recipients: tuple[str, ...] = ()
    cc: tuple[str, ...] = ()
    text: str
    labels: PageLabels


class ScoreAnswer(_Frozen):
    type: Literal["score"] = "score"
    name: str
    score: float
    confidence: float
    probabilities: tuple[float, ...]


class PredicateAnswer(_Frozen):
    type: Literal["predicate"] = "predicate"
    name: str
    probability: float


class RefusalAnswer(_Frozen):
    type: Literal["refusal"] = "refusal"
    name: str


Answer = Annotated[
    ScoreAnswer | PredicateAnswer | RefusalAnswer,
    Field(discriminator="type"),
]


class TokenUsage(_Frozen):
    input: int = 0
    cached: int = 0
    cache_write: int = 0
    output: int = 0
    reasoning: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input=self.input + other.input,
            cached=self.cached + other.cached,
            cache_write=self.cache_write + other.cache_write,
            output=self.output + other.output,
            reasoning=self.reasoning + other.reasoning,
        )


class BilledSpend(_Frozen):
    usage: TokenUsage = TokenUsage()
    cost_usd: float = 0.0

    def __add__(self, other: BilledSpend) -> BilledSpend:
        return BilledSpend(
            usage=self.usage + other.usage, cost_usd=self.cost_usd + other.cost_usd
        )


class ClassifierResult(_Frozen):
    engine: str
    model: str
    answers: dict[str, Answer] = Field(default_factory=dict)
    agent_disposition: Route | None = None
    agent_redact: bool | None = None
    usage: TokenUsage = TokenUsage()
    latency_s: float = 0.0
    cost_usd: float = 0.0
    error: str | None = None
    models: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    routing_tokens: int = 0
    parse_failed: bool = False


class RedactionResult(_Frozen):
    text: str
    model: str
    usage: TokenUsage = TokenUsage()
    latency_s: float = 0.0
    cost_usd: float = 0.0


class PageResult(_Frozen):
    doc_id: str
    decision: Literal["produce", "withhold"]
    route: Route
    redacted: bool = False
    produced_text: str | None = None
    escalated: bool = False
    attorney_decision: Literal["produce", "withhold"] | None = None
    classification: ClassifierResult | None = None
    redaction: RedactionResult | None = None
    failed_redaction: BilledSpend = BilledSpend()
    reason: str = ""
