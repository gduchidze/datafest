"""Wrappers that compose classifiers: a code pre-lookup and a second-opinion cascade."""

from __future__ import annotations

from doc_review_opt.agent.protocols import Classifier
from doc_review_opt.agent.routing import DEFAULT_POLICY, RoutingPolicy, decide
from doc_review_opt.agent.tools import check_prior_production
from doc_review_opt.models import ClassifierResult, PredicateAnswer, ReviewPage

RESPONSIVE_BAND_REASONS = frozenset(
    {"ambiguous_responsiveness", "low_responsive_confidence"}
)
PRIOR_TOOL = "check_prior_production"


def in_ambiguous_band(result: ClassifierResult, policy: RoutingPolicy) -> bool:
    return decide(result, policy).reason in RESPONSIVE_BAND_REASONS


def _probability(result: ClassifierResult, name: str) -> float:
    answer = result.answers.get(name)
    return answer.probability if isinstance(answer, PredicateAnswer) else 0.0


def _models(result: ClassifierResult) -> tuple[str, ...]:
    return result.models or (result.model,)


class PriorProductionGate:
    engine = "code"

    def __init__(self, inner: Classifier) -> None:
        self._inner = inner

    async def classify(self, page: ReviewPage, protocol: str) -> ClassifierResult:
        match = check_prior_production(page.doc_id)
        if not match["found"]:
            return await self._inner.classify(page, protocol)
        produced = match["status"] == "produced"
        return ClassifierResult(
            engine=self.engine,
            model="code",
            agent_disposition="withhold" if produced else "attorney_review",
            tools=(PRIOR_TOOL,),
        )


class CascadeClassifier:
    engine = "cascade"

    def __init__(
        self,
        primary: Classifier,
        second_opinion: Classifier,
        policy: RoutingPolicy = DEFAULT_POLICY,
    ) -> None:
        self._primary = primary
        self._second = second_opinion
        self._policy = policy

    async def classify(self, page: ReviewPage, protocol: str) -> ClassifierResult:
        first = await self._primary.classify(page, protocol)
        if not in_ambiguous_band(first, self._policy):
            return first
        second = await self._second.classify(page, protocol)
        chosen = first if second.error is not None else second
        answers = dict(chosen.answers)
        for name in ("privileged", "contains_pii"):
            if name in answers:
                answers[name] = PredicateAnswer(
                    name=name,
                    probability=max(
                        _probability(first, name), _probability(second, name)
                    ),
                )
        return chosen.model_copy(
            update={
                "answers": answers,
                "engine": self.engine,
                "models": tuple(dict.fromkeys(_models(first) + _models(second))),
                "usage": first.usage + second.usage,
                "cost_usd": first.cost_usd + second.cost_usd,
                "latency_s": first.latency_s + second.latency_s,
            }
        )
