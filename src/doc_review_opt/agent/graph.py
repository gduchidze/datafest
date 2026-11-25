"""The review graph: classify each page, route it, and finish in one terminal node."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import (
    Any,
    Literal,
    cast,
)

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Send
from pydantic import BaseModel

from doc_review_opt import models as review_models
from doc_review_opt.agent.nodes import (
    _PageNodes,
)
from doc_review_opt.agent.protocols import Classifier, Redactor
from doc_review_opt.agent.routing import (
    DEFAULT_POLICY,
    RoutingPolicy,
)
from doc_review_opt.agent.state import (
    BatchState,
    InterruptPayload,
    PageState,
)
from doc_review_opt.fixtures import PROTOCOL
from doc_review_opt.models import (
    Disposition,
    PageResult,
    ReviewPage,
    Route,
)

ReviewGraph = CompiledStateGraph[Any, Any, Any, Any]


STATE_MODELS = [
    (review_models.__name__, name)
    for name, value in vars(review_models).items()
    if isinstance(value, type)
    and issubclass(value, BaseModel)
    and value.__module__ == review_models.__name__
]


def review_checkpointer() -> InMemorySaver:
    return InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=STATE_MODELS))


def _pick_route(state: PageState) -> Route:
    return state["route"]


def _compile_page_graph(nodes: _PageNodes) -> Any:
    page_graph = StateGraph(PageState)
    page_graph.add_node("classify", nodes.classify)
    page_graph.add_node("attorney_review", nodes.attorney_review)
    page_graph.add_node("redact", nodes.redact)
    page_graph.add_node("produce", nodes.produce)
    page_graph.add_node("withhold", nodes.withhold)
    page_graph.add_edge(START, "classify")
    page_graph.add_conditional_edges(
        "classify", _pick_route, ["attorney_review", "redact", "produce", "withhold"]
    )
    for node in ("attorney_review", "redact", "produce", "withhold"):
        page_graph.add_edge(node, END)
    return page_graph.compile()


def build_review_graph(
    classifier: Classifier,
    redactor: Redactor,
    checkpointer: BaseCheckpointSaver[Any],
    *,
    policy: RoutingPolicy = DEFAULT_POLICY,
    protocol: str = PROTOCOL,
) -> ReviewGraph:

    def fan_out(state: BatchState) -> list[Send]:
        return [
            Send(
                "review_page",
                {"doc_id": page.doc_id, "page": page, "protocol": protocol},
            )
            for page in state["pages"]
        ]

    batch = StateGraph(BatchState)
    batch.add_node(
        "review_page", _compile_page_graph(_PageNodes(classifier, redactor, policy))
    )
    batch.add_conditional_edges(START, fan_out, ["review_page"])
    batch.add_edge("review_page", END)
    return batch.compile(checkpointer=checkpointer)


@dataclass(frozen=True)
class BatchOutcome:
    results: tuple[PageResult, ...]
    interrupts: tuple[InterruptPayload, ...]


async def run_batch(
    graph: ReviewGraph,
    pages: Sequence[ReviewPage],
    *,
    thread_id: str,
    attorney_decisions: Mapping[str, Disposition],
    default: Literal["produce", "withhold"] = "withhold",
    max_concurrency: int | None = None,
) -> BatchOutcome:
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    if max_concurrency is not None:
        config["max_concurrency"] = max_concurrency
    state = await graph.ainvoke({"pages": list(pages), "results": []}, config)
    payloads: list[InterruptPayload] = []
    while state.get("__interrupt__"):
        resume: dict[str, Disposition] = {}
        for pending in state["__interrupt__"]:
            payloads.append(pending.value)
            resume[pending.id] = attorney_decisions.get(
                pending.value["doc_id"], default
            )
        state = await graph.ainvoke(Command(resume=resume), config)
    return BatchOutcome(
        results=tuple(cast(list[PageResult], state["results"])),
        interrupts=tuple(payloads),
    )
