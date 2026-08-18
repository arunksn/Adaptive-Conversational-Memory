from dataclasses import dataclass

import pytest

from experiments.adaptive_memory import (
    AdaptiveMemory,
    AdaptiveMemoryResult,
)


@dataclass
class FakeMemory:
    memory_id: str
    content: str = "memory"


@dataclass
class FakeRetrievalResult:
    memory_id: str
    item: FakeMemory | None = None
    metadata: dict | None = None


@dataclass
class FakeRoutingResult:
    routes: list


class FakeAdaptiveRetriever:

    def __init__(self):
        self.calls = []

    def retrieve(
        self,
        query,
        top_k=5,
        start_time=None,
        end_time=None,
        procedure_id=None,
        state_id=None,
    ):
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "start_time": start_time,
                "end_time": end_time,
                "procedure_id": procedure_id,
                "state_id": state_id,
            }
        )

        if "programming language" in query.lower():

            return (
                FakeRoutingResult(
                    routes=["semantic"]
                ),
                [
                    FakeRetrievalResult(
                        memory_id="memory-python"
                    ),
                    FakeRetrievalResult(
                        memory_id="memory-other"
                    ),
                ],
            )

        return (
            FakeRoutingResult(
                routes=[]
            ),
            [],
        )


class FakeContextBuilder:

    def __init__(self):
        self.calls = []

    def build(
        self,
        results,
        query=None,
    ):
        self.calls.append(
            {
                "results": results,
                "query": query,
            }
        )

        return "fake-context"


def test_initialization():

    retriever = FakeAdaptiveRetriever()

    system = AdaptiveMemory(
        retriever
    )

    assert (
        system.adaptive_retriever
        is retriever
    )


def test_none_adaptive_retriever_rejected():

    with pytest.raises(ValueError):

        AdaptiveMemory(
            None
        )


def test_retrieve():

    retriever = FakeAdaptiveRetriever()

    system = AdaptiveMemory(
        retriever
    )

    result = system.retrieve(
        query="What programming language do I prefer?",
        k=2,
    )

    assert isinstance(
        result,
        AdaptiveMemoryResult,
    )

    assert result.retrieved_ids == [
        "memory-python",
        "memory-other",
    ]

    assert result.result_count == 2

    assert len(
        result.results
    ) == 2

    assert result.routing.routes == [
        "semantic"
    ]

    assert retriever.calls == [
        {
            "query":
                "What programming language do I prefer?",
            "top_k": 2,
            "start_time": None,
            "end_time": None,
            "procedure_id": None,
            "state_id": None,
        }
    ]


def test_retrieve_limits_results_to_k():

    retriever = FakeAdaptiveRetriever()

    system = AdaptiveMemory(
        retriever
    )

    result = system.retrieve(
        query="What programming language do I prefer?",
        k=1,
    )

    assert result.retrieved_ids == [
        "memory-python"
    ]

    assert result.result_count == 1

    assert len(
        result.results
    ) == 1


def test_retrieve_empty_query_rejected():

    system = AdaptiveMemory(
        FakeAdaptiveRetriever()
    )

    with pytest.raises(ValueError):

        system.retrieve(
            query="   ",
            k=2,
        )


def test_retrieve_invalid_k_rejected():

    system = AdaptiveMemory(
        FakeAdaptiveRetriever()
    )

    with pytest.raises(ValueError):

        system.retrieve(
            query="What do I prefer?",
            k=0,
        )


def test_retrieve_forwards_temporal_parameters():

    retriever = FakeAdaptiveRetriever()

    system = AdaptiveMemory(
        retriever
    )

    system.retrieve(
        query="What happened recently?",
        k=3,
        start_time="START",
        end_time="END",
        procedure_id="procedure-1",
        state_id="state-1",
    )

    assert retriever.calls == [
        {
            "query":
                "What happened recently?",
            "top_k": 3,
            "start_time": "START",
            "end_time": "END",
            "procedure_id": "procedure-1",
            "state_id": "state-1",
        }
    ]


def test_extract_memory_ids():

    results = [
        FakeRetrievalResult(
            memory_id="memory-1"
        ),
        FakeRetrievalResult(
            memory_id="memory-2"
        ),
    ]

    ids = AdaptiveMemory._extract_memory_ids(
        results
    )

    assert ids == [
        "memory-1",
        "memory-2",
    ]


def test_extract_memory_id_from_item():

    result = FakeRetrievalResult(
        memory_id=None,
        item=FakeMemory(
            memory_id="memory-item"
        ),
    )

    assert (
        AdaptiveMemory._extract_memory_id(
            result
        )
        == "memory-item"
    )


def test_extract_memory_id_from_dict():

    result = {
        "memory_id": "memory-dict"
    }

    assert (
        AdaptiveMemory._extract_memory_id(
            result
        )
        == "memory-dict"
    )


def test_extract_memory_id_from_item_dict():

    result = {
        "item": {
            "memory_id": "memory-item-dict"
        }
    }

    assert (
        AdaptiveMemory._extract_memory_id(
            result
        )
        == "memory-item-dict"
    )


def test_extract_memory_id_from_metadata():

    result = FakeRetrievalResult(
        memory_id=None,
        metadata={
            "memory_id":
                "memory-metadata"
        },
    )

    assert (
        AdaptiveMemory._extract_memory_id(
            result
        )
        == "memory-metadata"
    )


def test_extract_memory_ids_ignores_missing_ids():

    results = [
        FakeRetrievalResult(
            memory_id="memory-1"
        ),
        object(),
        {
            "content": "memory without id"
        },
    ]

    ids = AdaptiveMemory._extract_memory_ids(
        results
    )

    assert ids == [
        "memory-1"
    ]


def test_build_context_without_builder_rejected():

    system = AdaptiveMemory(
        FakeAdaptiveRetriever()
    )

    with pytest.raises(ValueError):

        system.build_context(
            query="What do I prefer?",
            context_builder=None,
        )


def test_build_context_with_builder():

    retriever = FakeAdaptiveRetriever()

    builder = FakeContextBuilder()

    system = AdaptiveMemory(
        retriever
    )

    context = system.build_context(
        query="What programming language do I prefer?",
        context_builder=builder,
        k=2,
    )

    assert context == "fake-context"

    assert len(
        builder.calls
    ) == 1

    assert (
        builder.calls[0]["query"]
        == "What programming language do I prefer?"
    )

    assert len(
        builder.calls[0]["results"]
    ) == 2


def test_run_without_context_builder():

    retriever = FakeAdaptiveRetriever()

    system = AdaptiveMemory(
        retriever
    )

    result = system.run(
        query="What programming language do I prefer?",
        k=2,
    )

    assert result["query"] == (
        "What programming language do I prefer?"
    )

    assert isinstance(
        result["retrieval"],
        AdaptiveMemoryResult,
    )

    assert result["retrieved_ids"] == [
        "memory-python",
        "memory-other",
    ]

    assert result["routing"].routes == [
        "semantic"
    ]

    assert result["context"] is None


def test_run_with_context_builder():

    retriever = FakeAdaptiveRetriever()

    builder = FakeContextBuilder()

    system = AdaptiveMemory(
        retriever
    )

    result = system.run(
        query="What programming language do I prefer?",
        k=2,
        context_builder=builder,
    )

    assert result["retrieved_ids"] == [
        "memory-python",
        "memory-other",
    ]

    assert result["context"] == (
        "fake-context"
    )

    assert len(
        builder.calls
    ) == 1